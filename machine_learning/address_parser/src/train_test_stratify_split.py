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
    s = to_str(v)  # <- use the coerced string
    return s.split() if s else []


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


def _front_order_realistic(fields: dict, rng: np.random.Generator, property_type: str = "") -> list[str]:
    """
    Choose the order for the premise+thoroughfare (front) in a realistic way.
    Uses presence of SubBuildingName / BuildingName / BuildingNumber / StreetName.
    """
    has_sub = bool(fields["SubBuildingName"])
    has_bnm = bool(fields["BuildingName"])
    has_bno = bool(fields["BuildingNumber"])
    has_str = bool(fields["StreetName"])
    is_flat = (to_str(property_type, upper=True) == "F") or has_sub

    # Start with nothing; we append labels that are actually present.
    seq: list[str] = []

    # --- Flats / apartments ---
    if is_flat and has_str:
        # Common: "FLAT X 10 THE MEWS HIGH STREET"
        patterns = [
            ["SubBuildingName", "BuildingNumber", "BuildingName", "StreetName"],  # very common
            ["SubBuildingName", "BuildingName", "BuildingNumber", "StreetName"],
            ["BuildingNumber", "BuildingName", "SubBuildingName", "StreetName"],
            ["BuildingName", "BuildingNumber", "SubBuildingName", "StreetName"],
            ["SubBuildingName", "StreetName", "BuildingNumber", "BuildingName"],  # occasional oddity
            ["BuildingNumber", "StreetName", "SubBuildingName", "BuildingName"],  # number before street
        ]
        probs = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.07], dtype=float)
        idx = rng.choice(len(patterns), p=probs / probs.sum())
        seq = [l for l in patterns[idx] if fields[l]]

    # --- Houses / non-flats with BOTH name & number ---
    elif has_bnm and has_bno and has_str:
        patterns = [
            ["BuildingNumber", "BuildingName", "StreetName"],  # "12 ROSE COTTAGE HIGH STREET"
            ["BuildingName", "BuildingNumber", "StreetName"],  # "ROSE COTTAGE 12 HIGH STREET"
            ["BuildingNumber", "StreetName", "BuildingName"],  # "12 HIGH STREET ROSE COTTAGE"
            ["BuildingName", "StreetName", "BuildingNumber"],  # less common
        ]
        probs = np.array([0.40, 0.35, 0.20, 0.05], dtype=float)
        idx = rng.choice(len(patterns), p=probs / probs.sum())
        seq = [l for l in patterns[idx] if fields[l]]

    # --- Number only ---
    elif has_bno and has_str:
        if rng.random() < 0.9:
            seq = ["BuildingNumber", "StreetName"]
        else:
            seq = ["StreetName", "BuildingNumber"]  # rare

    # --- Name only ---
    elif has_bnm and has_str:
        if rng.random() < 0.85:
            seq = ["BuildingName", "StreetName"]
        else:
            seq = ["StreetName", "BuildingName"]

    # --- No street (rural oddities) ---
    else:
        # Whatever we have, in a stable order
        for l in ["SubBuildingName", "BuildingName", "BuildingNumber", "StreetName"]:
            if fields[l]:
                seq.append(l)

    # Deduplicate and ensure only present labels remain
    seen = set()
    out = []
    for l in seq:
        if l not in seen and fields[l]:
            out.append(l);
            seen.add(l)
    return out


def _back_order_realistic(fields: dict, rng: np.random.Generator) -> list[str]:
    """
    Order the locality/town/postcode block realistically.
    - Postcode last
    - Locality usually before TownName, but sometimes after
    """
    has_loc = bool(fields["Locality"])
    has_twn = bool(fields["TownName"])
    order = []

    if has_loc and has_twn:
        if rng.random() < 0.9:
            order = ["Locality", "TownName"]
        else:
            order = ["TownName", "Locality"]
    elif has_twn:
        order = ["TownName"]
    elif has_loc:
        order = ["Locality"]
    else:
        order = []

    # Postcode at the end if present
    if fields["Postcode"]:
        order.append("Postcode")
    return order


def build_address_element(row: dict, rng: np.random.Generator, swap_locality_prob: float = 0.0) -> etree._Element:
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
    # Front (premises + thoroughfare)
    front = _front_order_realistic(fields, rng, property_type=row.get("property_type", ""))
    # Back (locality/town/postcode)
    back = _back_order_realistic(fields, rng)

    for label in front + back:
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
