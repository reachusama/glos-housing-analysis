# address_to_xml.py
import re
import pandas as pd
from lxml import etree

LABELS = [
    'OrganisationName', 'DepartmentName', 'SubBuildingName', 'BuildingName',
    'BuildingNumber', 'StreetName', 'Locality', 'TownName', 'Postcode'
]

# UK postcode (split outcode / incode)
POSTCODE_RE = re.compile(
    r"""^\s*(
            (?:GIR\s?0AA)|
            (?:[A-PR-UWYZ][0-9][0-9A-HJKMNPR-Y]?)|
            (?:[A-PR-UWYZ][A-HK-Y][0-9][0-9A-HJKMNPR-Y]?)
        )\s*([0-9][ABD-HJLNP-UW-Z]{2})\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

FLAT_WORDS = {"FLAT", "FLT", "APARTMENT", "APPT", "APT", "APPTS", "ROOM", "ANNEX", "ANNEXE", "UNIT", "BLOCK", "BLK"}


def tokenize(text: str):
    if not isinstance(text, str): return []
    text = text.strip()
    if not text: return []
    return [t for t in re.split(r"\s+", text) if t]


def split_postcode(pc: str):
    if not isinstance(pc, str): return []
    pc = pc.strip().upper()
    m = POSTCODE_RE.match(pc)
    if m:
        return [m.group(1).upper(), m.group(2).upper()]
    parts = pc.split()
    return parts if len(parts) == 2 else ([pc] if pc else [])


def split_paon(paon_raw: str):
    """
    Return (building_number_tokens, building_name_tokens) from PAON.
    Rules:
      - purely numeric/alphanumeric token(s) -> BuildingNumber
      - remaining word tokens -> BuildingName
    """
    toks = tokenize(paon_raw)
    if not toks: return [], []
    # If every token is alnum-with-digits (e.g., 12, 12A, 3-5), treat all as BuildingNumber
    has_word = any(re.search(r"[A-Za-z]", t) and not re.search(r"\d", t) for t in toks)
    has_digit = any(re.search(r"\d", t) for t in toks)

    # Extract digit-containing tokens as number; rest as name
    number_tokens = [t for t in toks if re.search(r"\d", t)]
    name_tokens = [t for t in toks if t not in number_tokens]

    # If no digits at all, it's a pure building name
    if not has_digit:
        return [], toks
    # If tokens are a mix (e.g., "10 THE MEWS")
    return number_tokens, name_tokens


def saon_to_subbuilding(saon_raw: str, paon_raw: str = ""):
    """
    Build SubBuildingName tokens.
    If SAON empty but PAON contains 'FLAT' etc., treat that part as sub-building.
    """
    saon_toks = tokenize(saon_raw)
    if saon_toks:
        return saon_toks

    paon_toks = tokenize(paon_raw)
    if paon_toks and (paon_toks[0].upper() in FLAT_WORDS):
        # e.g., "Flat 2 10 The Mews" -> SubBuildingName = ["Flat","2"]
        out = []
        for t in paon_toks:
            if t.upper() in FLAT_WORDS or re.fullmatch(r"\d+[A-Za-z]?", t):
                out.append(t)
            else:
                break
        return out
    return []


def row_to_addressstring(row):
    addr = etree.Element("AddressString")

    # SAON/SOAN handling
    saon_col = "SAON" if "SAON" in row.index else ("SOAN" if "SOAN" in row.index else None)
    saon = row[saon_col] if saon_col else ""
    paon = row.get("PAON", "")

    subbuilding_tokens = saon_to_subbuilding(saon, paon)
    for t in subbuilding_tokens:
        etree.SubElement(addr, "SubBuildingName").text = t

    # PAON split into BuildingNumber/BuildingName (excluding any leading subbuilding we already used)
    paon_remaining = " ".join([t for t in tokenize(paon) if t not in subbuilding_tokens]) if paon else ""
    bn_tokens, bname_tokens = split_paon(paon_remaining)
    for t in bname_tokens:
        etree.SubElement(addr, "BuildingName").text = t
    for t in bn_tokens:
        etree.SubElement(addr, "BuildingNumber").text = t

    # Street
    for t in tokenize(row.get("street", "")):
        etree.SubElement(addr, "StreetName").text = t

    # Locality
    for t in tokenize(row.get("locality", "")):
        etree.SubElement(addr, "Locality").text = t

    # Town
    for t in tokenize(row.get("townname", "")):
        etree.SubElement(addr, "TownName").text = t

    # Postcode -> outcode + incode
    for t in split_postcode(row.get("postcode", "")):
        etree.SubElement(addr, "Postcode").text = t

    # NOTE: district/county intentionally omitted

    return addr


def dataframe_to_xml(df: pd.DataFrame):
    root = etree.Element("AddressCollection")
    for _, row in df.iterrows():
        root.append(row_to_addressstring(row))
    return etree.tostring(root, encoding="utf-8", pretty_print=True)
