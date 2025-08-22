# balanced_streaming_xml_compat.py
from pathlib import Path
from typing import Optional, Dict, List
import re
import numpy as np
import pandas as pd
from lxml import etree
import gzip

# ---------- token + postcode helpers ----------
FLAT_WORDS = {"FLAT", "APARTMENT", "APT", "APPTS", "ROOM", "UNIT", "ANNEX", "ANNEXE", "BLOCK", "BLK", "STUDIO"}

# Put this near the top of your module
LABELS = [
    "SubBuildingName",
    "BuildingName",
    "BuildingNumber",
    "StreetName",
    "Locality",
    "TownName",
    "Postcode",
]


def choose_order_random(
        present_labels: set,
        rng: np.random.Generator,
        *,
        postcode_last: bool = True,
        smart_pair_prob: float = 0.7,  # try to keep Street next to a Building* label this often
) -> list[str]:
    """
    Random label order for the current row:
      - shuffles all present labels
      - keeps Postcode at the end (if postcode_last=True)
      - with probability smart_pair_prob, makes StreetName adjacent to BuildingNumber/BuildingName
    """
    # start from your canonical label list, keep only the ones that are present
    pool = [lab for lab in LABELS if lab in present_labels]

    # optionally pop postcode out to re-append later
    has_pc = ("Postcode" in pool)
    if postcode_last and has_pc:
        pool.remove("Postcode")

    # random permutation
    order = rng.permutation(pool).tolist()

    # "smart" tweak: keep StreetName next to a building component sometimes
    if "StreetName" in order and smart_pair_prob and rng.random() < smart_pair_prob:
        # pick a target to sit next to: prefer BuildingNumber, else BuildingName
        target = None
        if "BuildingNumber" in order and "BuildingName" in order:
            target = "BuildingNumber" if rng.random() < 0.6 else "BuildingName"
        elif "BuildingNumber" in order:
            target = "BuildingNumber"
        elif "BuildingName" in order:
            target = "BuildingName"

        if target is not None:
            # move StreetName right after the chosen target
            try:
                order.remove("StreetName")
                ti = order.index(target)
                order.insert(ti + 1, "StreetName")
            except ValueError:
                # target vanished somehow; ignore
                pass

    if postcode_last and has_pc:
        order.append("Postcode")

    return order


def to_str(v, upper: bool = False) -> str:
    """
    Coerce any scalar (incl. NaN/None/ints/floats) to a clean string.
    - NaN/None -> ""
    - 12      -> "12"
    - 12.0    -> "12"
    - trims whitespace, optional .upper()
    """
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return ""
    # normal string-ish
    s = str(v).strip()
    # collapse floats like "12.0" to "12"
    s = re.sub(r"^(\d+)\.0$", r"\1", s)
    return s.upper() if upper else s


def tokens_from(v: Optional[str]) -> List[str]:
    if not to_str(v):
        return []
    v = v.strip()
    return v.split() if v else []


def outcode_incode_from(row) -> list[str]:
    oc = to_str(row.get("Outcode"), upper=True)
    ic = to_str(row.get("Incode"), upper=True)
    if oc and ic:
        return [oc, ic]
    pc = to_str(row.get("Postcode"), upper=True)
    if not pc:
        return []
    parts = pc.split()
    return parts if len(parts) == 2 else [pc]


# ---------- pattern labelling (for balanced sampling) ----------
_num_only_re = re.compile(r"^\d+$")
_alnum_re = re.compile(r"^(?:\d+[A-Za-z]|[A-Za-z]\d+)\w*$")
_flat_start_re = re.compile(r"^\s*(" + "|".join(sorted(FLAT_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def paon_pattern(bname, bnum) -> str:
    has_name = bool(to_str(bname))
    has_num = bool(to_str(bnum))
    if has_name and has_num: return "name+number"
    if has_num:              return "number"
    if has_name:             return "name"
    return "none"


def saon_pattern(saon) -> str:
    s = to_str(saon)
    if not s:                       return "none"
    if _flat_start_re.search(s):    return "flat"
    if _num_only_re.fullmatch(s):   return "num"
    if _alnum_re.fullmatch(s):      return "alnum"
    return "name"


def make_strata_key(row: pd.Series, include_property_type: bool = True) -> str:
    paon = paon_pattern(row.get("BuildingName"), row.get("BuildingNumber"))
    saon = saon_pattern(row.get("SubBuildingName"))
    if include_property_type:
        pt = (row.get("property_type") or "").strip().upper() or "NA"
        return f"{paon}|{saon}|{pt}"
    return f"{paon}|{saon}"


# ---------- balanced sampling ----------
def balanced_sample(
        df: pd.DataFrame,
        n_total: int,
        seed: int = 42,
        include_property_type_in_strata: bool = True,
) -> pd.DataFrame:
    if n_total is None or n_total >= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)
    strata = df.apply(lambda r: make_strata_key(r, include_property_type_in_strata), axis=1)
    df2 = df.copy()
    df2["_stratum"] = strata

    groups = df2.groupby("_stratum", sort=False)
    keys = list(groups.groups.keys())
    if not keys:
        return df.head(n_total).copy()

    base_quota = max(1, n_total // len(keys))
    chosen_idx = []
    budget = n_total

    # first pass
    for key in keys:
        g = groups.get_group(key)
        take = min(base_quota, len(g))
        if take:
            chosen_idx += g.sample(n=take, random_state=int(rng.integers(0, 1_000_000))).index.tolist()
            budget -= take

    # second pass
    if budget > 0:
        leftovers = []
        taken = set(chosen_idx)
        for key in keys:
            g = groups.get_group(key)
            rem = g.drop(index=[i for i in g.index if i in taken])
            if len(rem):
                leftovers.append(rem)
        if leftovers:
            pool = pd.concat(leftovers, axis=0)
            add = min(budget, len(pool))
            if add:
                chosen_idx += pool.sample(n=add, random_state=int(rng.integers(0, 1_000_000))).index.tolist()

    sampled = (
        df2.loc[chosen_idx]
        .drop(columns=["_stratum"])
        .sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        .reset_index(drop=True)
    )
    if len(sampled) > n_total:
        sampled = sampled.head(n_total)
    return sampled


def build_address_element(row: dict, rng: np.random.Generator, swap_locality_prob: float = 0.0) -> etree._Element:
    """
    Build one <AddressString> with a fully randomised label order.
    Note: swap_locality_prob kept for API compatibility but no longer needed
          because Locality/TownName are part of the global shuffle now.
    """
    el = etree.Element("AddressString")

    fields = {
        "SubBuildingName": tokens_from(row.get("SubBuildingName")),
        "BuildingName": tokens_from(row.get("BuildingName")),
        "BuildingNumber": tokens_from(row.get("BuildingNumber")),
        "StreetName": tokens_from(row.get("StreetName")),
        "Locality": tokens_from(row.get("Locality")),
        "TownName": tokens_from(row.get("TownName")),
        "Postcode": outcode_incode_from(row),
    }

    present = {k for k, toks in fields.items() if toks}
    # Random order across ALL present labels, keeping postcode last
    order = choose_order_random(present, rng, postcode_last=True, smart_pair_prob=0.7)

    for label in order:
        for tok in fields[label]:
            etree.SubElement(el, label).text = tok

    return el


# ---------- streaming writer (no xmlfile.element / .start) ----------
def write_train_test_streaming(
        df: pd.DataFrame,
        output_dir: str,
        n_total: Optional[int] = None,
        test_ratio: float = 0.10,
        rows_per_shard: int = 200_000,
        seed: int = 42,
        swap_locality_prob: float = 0.25,
        include_property_type_in_strata: bool = True,
        train_prefix: str = "training",
        holdout_prefix: str = "holdout",
        compress: bool = False,  # NEW: write .xml.gz if True
) -> Dict[str, object]:
    """
    Balanced sample → stream to sharded train/holdout XML files using plain file I/O.
    Returns: {'counts': {'train': int, 'test': int}, 'files': {'train': [..], 'test': [..]}}
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir);
    out_dir.mkdir(parents=True, exist_ok=True)

    def has_sig(s):
        return isinstance(s, str) and s.strip() != ""

    mask = (
            df["BuildingNumber"].apply(has_sig) |
            df["BuildingName"].apply(has_sig) |
            df["StreetName"].apply(has_sig)
    )
    base = df.loc[mask].reset_index(drop=True)

    data = balanced_sample(
        base,
        n_total=n_total if (n_total is not None and n_total > 0) else len(base),
        seed=seed,
        include_property_type_in_strata=include_property_type_in_strata,
    ).reset_index(drop=True)

    files = {"train": [], "test": []}
    counts = {"train": 0, "test": 0}
    shard_counts = {"train": 0, "test": 0}
    shard_idx = {"train": 0, "test": 0}
    writers: Dict[str, Optional[object]] = {"train": None, "test": None}  # file handles

    def _open(kind: str):
        prefix = train_prefix if kind == "train" else holdout_prefix
        ext = ".xml.gz" if compress else ".xml"
        fname = out_dir / f"{prefix}_{shard_idx[kind]:04d}{ext}"
        fh = gzip.open(fname, "wb") if compress else open(fname, "wb")

        fh.write(b"<AddressCollection>\n")
        writers[kind] = fh
        shard_counts[kind] = 0
        files[kind].append(str(fname))

    def _close(kind: str):
        fh = writers[kind]
        if fh is not None:
            fh.write(b"</AddressCollection>\n")
            fh.close()
            writers[kind] = None

    _open("train");
    _open("test")
    cols = list(data.columns)

    for row in data.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(cols, tuple(row)))
        kind = "test" if rng.random() < test_ratio else "train"

        # rollover shard
        if shard_counts[kind] >= rows_per_shard:
            _close(kind)
            shard_idx[kind] += 1
            _open(kind)

        # build one small element → serialise → write
        addr_el = build_address_element(row_dict, rng, swap_locality_prob=swap_locality_prob)
        xml_bytes = etree.tostring(addr_el, encoding="utf-8", with_tail=False)
        writers[kind].write(xml_bytes + b"\n")

        shard_counts[kind] += 1
        counts[kind] += 1

    _close("train")
    _close("test")
    return {"counts": counts, "files": files}
