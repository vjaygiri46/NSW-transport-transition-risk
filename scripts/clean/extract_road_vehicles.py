"""Extract selected BITRE Road Vehicles tables into tidy CSV files.

This parser intentionally avoids openpyxl so it can run in the current local
environment. It reads XLSX files as zipped XML, locates sheets by name, detects
grouped table rows, and normalizes labels across releases.
"""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_raw" / "bitre" / "road_vehicles"
OUT_DIR = ROOT / "data_processed" / "road_vehicles"

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

JURISDICTIONS = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Western Australia": "WA",
    "Tasmania": "TAS",
    "Northern Territory": "NT",
    "Australian Capital Territory": "ACT",
    "Australia": "AUS",
}

MOTIVE_POWER = {
    "petrol": "petrol",
    "diesel": "diesel",
    "dual fuel": "dual_fuel",
    "dual fuel ": "dual_fuel",
    "hybrid electric": "hybrid_electric",
    "hevs": "hybrid_electric",
    "battery/fuel-cell electric": "battery_fuel_cell_electric",
    "battery/fuel cell electric": "battery_fuel_cell_electric",
    "bev/fcevs": "battery_fuel_cell_electric",
    "other": "other",
    "other /  not specified": "other_or_not_specified",
    "other / not specified": "other_or_not_specified",
    "not stated": "not_stated",
    "total": "total",
}


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_number(value: object) -> float | None:
    text = clean_text(value).replace(",", "")
    if not text or text.lower() in {"na", "n/a", "-", "—"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: object) -> int | None:
    num = safe_number(value)
    if num is None:
        return None
    return int(round(num))


def normalize_label(value: object) -> str:
    text = clean_text(value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def text_of(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    return [text_of(si) for si in root.findall("main:si", NS)]


def rel_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    mapping = {}
    for rel in rels.findall("rel:Relationship", NS):
        target = rel.attrib["Target"]
        if not target.startswith("/"):
            target = "xl/" + target
        mapping[rel.attrib["Id"]] = target.lstrip("/")
    return mapping


def sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    targets = rel_targets(zf)
    paths = {}
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        rid = sheet.attrib[f"{{{NS['office_rel']}}}id"]
        paths[sheet.attrib["name"]] = targets[rid]
    return paths


def col_to_int(col: str) -> int:
    value = 0
    for char in col:
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return value


def ref_to_row_col(ref: str) -> tuple[int, int]:
    match = re.match(r"([A-Za-z]+)(\d+)", ref)
    if not match:
        return 0, 0
    return int(match.group(2)), col_to_int(match.group(1))


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value = cell.find("main:v", NS)
    if cell_type == "s":
        raw = text_of(value)
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(shared):
                return shared[idx]
        return raw
    if cell_type == "inlineStr":
        return text_of(cell.find("main:is", NS))
    return text_of(value)


def read_sheet(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(sheet_path))
    output: list[list[str]] = []
    for row in root.findall("main:sheetData/main:row", NS):
        values_by_col: dict[int, str] = {}
        max_col = 0
        for cell in row.findall("main:c", NS):
            _, col = ref_to_row_col(cell.attrib.get("r", ""))
            max_col = max(max_col, col)
            values_by_col[col] = cell_value(cell, shared)
        output.append([values_by_col.get(i, "") for i in range(1, max_col + 1)])
    return output


def workbook_release_year(path: Path, rows: list[list[str]]) -> int:
    match = re.search(r"20\d{2}", path.name)
    if match:
        return int(match.group(0))
    for row in rows[:12]:
        joined = " ".join(clean_text(v) for v in row)
        match = re.search(r"January\s+(20\d{2})", joined)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer release year: {path}")


def row_nonempty(row: list[str]) -> list[str]:
    return [clean_text(v) for v in row if clean_text(v)]


def trim_leading_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    first_data_col: int | None = None
    for row in rows:
        for idx, value in enumerate(row):
            if clean_text(value):
                first_data_col = idx if first_data_col is None else min(first_data_col, idx)
                break
    if not first_data_col:
        return rows
    return [row[first_data_col:] if len(row) > first_data_col else [] for row in rows]


def is_year_cell(value: str) -> bool:
    text = clean_text(value)
    return bool(re.fullmatch(r"20\d{2}", text))


def find_header_row(rows: list[list[str]], required_tokens: set[str]) -> int:
    for idx, row in enumerate(rows):
        labels = {normalize_label(v) for v in row if clean_text(v)}
        has_year_column = any(label == "year" or label.startswith("year ") for label in labels)
        if has_year_column and required_tokens.intersection(labels):
            return idx
    raise ValueError(f"Could not find header row with tokens {required_tokens}")


def get_cell(row: list[str], idx: int) -> str:
    return row[idx] if idx < len(row) else ""


def extract_grouped_wide(
    *,
    workbook: Path,
    release_year: int,
    table_name: str,
    rows: list[list[str]],
    value_columns: dict[int, str],
    group_column_name: str,
    value_name: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current_group = ""
    for row in rows:
        first = clean_text(get_cell(row, 0))
        nonempty = row_nonempty(row)
        if not nonempty:
            continue
        if len(nonempty) == 1 and not is_year_cell(nonempty[0]):
            current_group = nonempty[0]
            continue
        if not current_group or not is_year_cell(first):
            continue
        year = int(first)
        for col_idx, variable in value_columns.items():
            value = safe_number(get_cell(row, col_idx))
            if value is None:
                continue
            records.append(
                {
                    "source_workbook": workbook.name,
                    "release_year": release_year,
                    "table": table_name,
                    group_column_name: current_group,
                    "year": year,
                    "variable": variable,
                    value_name: value,
                }
            )
    return records


def extract_table_1_3(
    workbook: Path, release_year: int, table_name: str, rows: list[list[str]], value_name: str
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows, {"new south wales", "australia"})
    header = rows[header_idx]
    value_columns = {
        idx: JURISDICTIONS[clean_text(value)]
        for idx, value in enumerate(header)
        if clean_text(value) in JURISDICTIONS
    }
    records = extract_grouped_wide(
        workbook=workbook,
        release_year=release_year,
        table_name=table_name,
        rows=rows[header_idx + 1 :],
        value_columns=value_columns,
        group_column_name="vehicle_type",
        value_name=value_name,
    )
    for record in records:
        record["jurisdiction"] = record.pop("variable")
    return records


def extract_table_4(
    workbook: Path, release_year: int, rows: list[list[str]]
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows, {"petrol", "diesel", "hevs", "hybrid electric"})
    header = rows[header_idx]
    value_columns = {}
    for idx, value in enumerate(header):
        label = normalize_label(value)
        if label in MOTIVE_POWER:
            value_columns[idx] = MOTIVE_POWER[label]
    records = extract_grouped_wide(
        workbook=workbook,
        release_year=release_year,
        table_name="Table 4",
        rows=rows[header_idx + 1 :],
        value_columns=value_columns,
        group_column_name="vehicle_type",
        value_name="vehicles",
    )
    for record in records:
        record["motive_power"] = record.pop("variable")
    return records


def extract_table_5(
    workbook: Path, release_year: int, rows: list[list[str]]
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows, {"petrol", "diesel", "hevs", "hybrid electric"})
    header = rows[header_idx]
    value_columns = {}
    for idx, value in enumerate(header):
        label = normalize_label(value)
        if label in MOTIVE_POWER:
            value_columns[idx] = MOTIVE_POWER[label]
    records = extract_grouped_wide(
        workbook=workbook,
        release_year=release_year,
        table_name="Table 5",
        rows=rows[header_idx + 1 :],
        value_columns=value_columns,
        group_column_name="jurisdiction_name",
        value_name="vehicles",
    )
    for record in records:
        name = str(record.pop("jurisdiction_name"))
        record["jurisdiction"] = JURISDICTIONS.get(name, name)
        record["motive_power"] = record.pop("variable")
    return records


def extract_table_6(
    workbook: Path, release_year: int, rows: list[list[str]]
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows, {"new south wales", "australia"})
    header = rows[header_idx]
    value_columns = {
        idx: JURISDICTIONS[clean_text(value)]
        for idx, value in enumerate(header)
        if clean_text(value) in JURISDICTIONS
    }
    records: list[dict[str, object]] = []
    current_vehicle_type = ""
    for row in rows[header_idx + 1 :]:
        nonempty = row_nonempty(row)
        if not nonempty:
            continue
        first = clean_text(get_cell(row, 0))
        if len(nonempty) == 1 and not re.search(r"\d", first):
            current_vehicle_type = nonempty[0]
            continue
        if not current_vehicle_type or not re.search(r"\d", first):
            continue
        for col_idx, jurisdiction in value_columns.items():
            value = safe_number(get_cell(row, col_idx))
            if value is None:
                continue
            records.append(
                {
                    "source_workbook": workbook.name,
                    "release_year": release_year,
                    "table": "Table 6",
                    "vehicle_type": current_vehicle_type,
                    "manufacture_period": first,
                    "jurisdiction": jurisdiction,
                    "vehicles": value,
                }
            )
    return records


def dedupe_latest(records: list[dict[str, object]], key_fields: list[str]) -> list[dict[str, object]]:
    latest: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        key = tuple(record[field] for field in key_fields)
        existing = latest.get(key)
        if existing is None or int(record["release_year"]) > int(existing["release_year"]):
            latest[key] = record
    return sorted(latest.values(), key=lambda r: tuple(str(r.get(f, "")) for f in key_fields))


def write_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    table1: list[dict[str, object]] = []
    table3: list[dict[str, object]] = []
    table4: list[dict[str, object]] = []
    table5: list[dict[str, object]] = []
    table6: list[dict[str, object]] = []

    for workbook in sorted(RAW_DIR.glob("*.xlsx")):
        with zipfile.ZipFile(workbook) as zf:
            shared = load_shared_strings(zf)
            paths = sheet_paths(zf)
            index_rows = read_sheet(zf, paths["Index"], shared)
            release_year = workbook_release_year(workbook, index_rows)
            sheets = {name: trim_leading_empty_columns(read_sheet(zf, path, shared)) for name, path in paths.items()}

        table1.extend(extract_table_1_3(workbook, release_year, "Table 1", sheets["Table 1"], "vehicles"))
        table3.extend(extract_table_1_3(workbook, release_year, "Table 3", sheets["Table 3"], "average_age_years"))
        table4.extend(extract_table_4(workbook, release_year, sheets["Table 4"]))
        table5.extend(extract_table_5(workbook, release_year, sheets["Table 5"]))
        table6.extend(extract_table_6(workbook, release_year, sheets["Table 6"]))

    write_csv(
        OUT_DIR / "vehicle_stock_by_type_all_releases.csv",
        table1,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "vehicles"],
    )
    write_csv(
        OUT_DIR / "vehicle_average_age_by_type_all_releases.csv",
        table3,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "average_age_years"],
    )
    write_csv(
        OUT_DIR / "motive_power_by_vehicle_type_all_releases.csv",
        table4,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "motive_power", "vehicles"],
    )
    write_csv(
        OUT_DIR / "motive_power_by_jurisdiction_all_releases.csv",
        table5,
        ["source_workbook", "release_year", "table", "jurisdiction", "year", "motive_power", "vehicles"],
    )
    write_csv(
        OUT_DIR / "manufacture_period_by_type_all_releases.csv",
        table6,
        [
            "source_workbook",
            "release_year",
            "table",
            "vehicle_type",
            "manufacture_period",
            "jurisdiction",
            "vehicles",
        ],
    )

    table5_latest = dedupe_latest(table5, ["jurisdiction", "year", "motive_power"])
    table4_latest = dedupe_latest(table4, ["vehicle_type", "year", "motive_power"])
    table1_latest = dedupe_latest(table1, ["vehicle_type", "year", "jurisdiction"])
    table3_latest = dedupe_latest(table3, ["vehicle_type", "year", "jurisdiction"])

    write_csv(
        OUT_DIR / "motive_power_by_jurisdiction_latest.csv",
        table5_latest,
        ["source_workbook", "release_year", "table", "jurisdiction", "year", "motive_power", "vehicles"],
    )
    write_csv(
        OUT_DIR / "motive_power_by_vehicle_type_latest.csv",
        table4_latest,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "motive_power", "vehicles"],
    )
    write_csv(
        OUT_DIR / "vehicle_stock_by_type_latest.csv",
        table1_latest,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "vehicles"],
    )
    write_csv(
        OUT_DIR / "vehicle_average_age_by_type_latest.csv",
        table3_latest,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "average_age_years"],
    )

    summary = [
        f"# Road Vehicles Extraction Summary",
        "",
        f"Workbooks processed: {len(list(RAW_DIR.glob('*.xlsx')))}",
        f"Table 1 records, all releases: {len(table1)}",
        f"Table 3 records, all releases: {len(table3)}",
        f"Table 4 records, all releases: {len(table4)}",
        f"Table 5 records, all releases: {len(table5)}",
        f"Table 6 records, all releases: {len(table6)}",
        "",
        f"Latest Table 5 records after overlap dedupe: {len(table5_latest)}",
        "",
        "Revision rule: overlapping year/category values keep the latest release year.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
