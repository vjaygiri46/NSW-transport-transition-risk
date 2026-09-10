"""Extract ABS Motor Vehicle Census XLS files into tidy CSV outputs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_raw" / "ABS"
OUT_DIR = ROOT / "data_processed" / "abs_motor_vehicle_census"

JURISDICTIONS = {
    "NEW SOUTH WALES": "NSW",
    "New South Wales": "NSW",
    "VICTORIA": "VIC",
    "Victoria": "VIC",
    "QUEENSLAND": "QLD",
    "Queensland": "QLD",
    "SOUTH AUSTRALIA": "SA",
    "South Australia": "SA",
    "WESTERN AUSTRALIA": "WA",
    "Western Australia": "WA",
    "TASMANIA": "TAS",
    "Tasmania": "TAS",
    "NORTHERN TERRITORY": "NT",
    "Northern Territory": "NT",
    "AUSTRALIAN CAPITAL TERRITORY": "ACT",
    "Australian Capital Territory": "ACT",
    "AUSTRALIA": "AUS",
    "Australia": "AUS",
}

FUEL_COLUMNS = {
    "Leaded": "petrol_leaded",
    "Unleaded": "petrol_unleaded",
    "Petrol": "petrol_total",
    "Total": "petrol_total",
    "Diesel": "diesel",
    "Other": "other",
    "All fuel types": "total",
}


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_number(value: object) -> float | None:
    text = clean_text(value).replace(",", "")
    if not text or text.lower() in {"na", "n/a", "-", "–", "—"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_year(value: object) -> bool:
    return bool(re.fullmatch(r"20\d{2}", clean_text(value)))


def release_year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        raise ValueError(f"Could not infer release year from {path.name}")
    return int(match.group(1))


def read_sheet(path: Path, sheet: str) -> list[list[object]]:
    df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object).where(pd.notna, "")
    return df.values.tolist()


def first_nonempty(row: list[object]) -> str:
    for value in row:
        text = clean_text(value)
        if text:
            return text
    return ""


def normalize_vehicle_type(value: str) -> str:
    text = clean_text(value).upper().replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    mapping = {
        "PASSENGER VEHICLES": "Passenger vehicles",
        "CAMPERVANS": "Campervans",
        "LIGHT COMMERCIAL VEHICLES": "Light commercial vehicles",
        "LIGHT RIGID TRUCKS": "Light rigid trucks",
        "HEAVY RIGID TRUCKS": "Heavy rigid trucks",
        "ARTICULATED TRUCKS": "Articulated trucks",
        "NON FREIGHT CARRYING VEHICLES": "Non-freight carrying vehicles",
        "BUSES": "Buses",
        "MOTOR CYCLES": "Motor cycles",
        "MOTORCYCLES": "Motor cycles",
        "TOTAL MOTOR VEHICLES": "Total motor vehicles",
    }
    return mapping.get(text, clean_text(value).title())


def find_jurisdiction_header(rows: list[list[object]]) -> int:
    for idx, row in enumerate(rows):
        labels = {clean_text(v) for v in row if clean_text(v)}
        if "New South Wales" in labels and "Australia" in labels:
            return idx
    raise ValueError("Could not find jurisdiction header")


def find_fuel_header(rows: list[list[object]]) -> int:
    for idx, row in enumerate(rows):
        labels = [clean_text(v) for v in row if clean_text(v)]
        if "Diesel" in labels and "Other" in labels and "All fuel types" in labels:
            return idx
    raise ValueError("Could not find fuel header")


def extract_grouped_jurisdiction_table(
    *,
    workbook: Path,
    sheet: str,
    table_name: str,
    value_name: str,
) -> list[dict[str, object]]:
    rows = read_sheet(workbook, sheet)
    header_idx = find_jurisdiction_header(rows)
    header = rows[header_idx]
    columns = {
        idx: JURISDICTIONS[clean_text(value)]
        for idx, value in enumerate(header)
        if clean_text(value) in JURISDICTIONS
    }

    records: list[dict[str, object]] = []
    current_vehicle_type = ""
    rel_year = release_year(workbook)
    for row in rows[header_idx + 1 :]:
        first = first_nonempty(row)
        if not first:
            continue
        if first.startswith("©"):
            break
        if is_year(first):
            if not current_vehicle_type:
                continue
            year = int(first)
            for col_idx, jurisdiction in columns.items():
                value = safe_number(row[col_idx] if col_idx < len(row) else "")
                if value is None:
                    continue
                records.append(
                    {
                        "source_workbook": workbook.name,
                        "release_year": rel_year,
                        "table": table_name,
                        "vehicle_type": current_vehicle_type,
                        "year": year,
                        "jurisdiction": jurisdiction,
                        value_name: value,
                    }
                )
        else:
            current_vehicle_type = normalize_vehicle_type(first)
    return records


def extract_fuel_table(
    *,
    workbook: Path,
    sheet: str,
    table_name: str,
    group_name: str,
) -> list[dict[str, object]]:
    rows = read_sheet(workbook, sheet)
    header_idx = find_fuel_header(rows)
    header = rows[header_idx]
    fuel_columns = {}
    for idx, value in enumerate(header):
        label = clean_text(value)
        if label in FUEL_COLUMNS:
            fuel_columns[idx] = FUEL_COLUMNS[label]

    records: list[dict[str, object]] = []
    current_group = ""
    rel_year = release_year(workbook)
    for row in rows[header_idx + 1 :]:
        first = first_nonempty(row)
        if not first:
            continue
        if first.startswith("©"):
            break
        if is_year(first):
            if not current_group:
                continue
            year = int(first)
            for col_idx, fuel_type in fuel_columns.items():
                value = safe_number(row[col_idx] if col_idx < len(row) else "")
                if value is None:
                    continue
                record = {
                    "source_workbook": workbook.name,
                    "release_year": rel_year,
                    "table": table_name,
                    "year": year,
                    "fuel_type": fuel_type,
                    "vehicles": value,
                }
                if group_name == "jurisdiction":
                    record["jurisdiction"] = JURISDICTIONS.get(current_group, current_group)
                else:
                    record["vehicle_type"] = normalize_vehicle_type(current_group)
                records.append(record)
        else:
            current_group = first
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

    for workbook in sorted(RAW_DIR.glob("*.xls")):
        table1.extend(
            extract_grouped_jurisdiction_table(
                workbook=workbook,
                sheet="Table_1",
                table_name="Table 1",
                value_name="vehicles",
            )
        )
        table3.extend(
            extract_grouped_jurisdiction_table(
                workbook=workbook,
                sheet="Table_3",
                table_name="Table 3",
                value_name="average_age_years",
            )
        )
        table4.extend(
            extract_fuel_table(
                workbook=workbook,
                sheet="Table_4",
                table_name="Table 4",
                group_name="vehicle_type",
            )
        )
        table5.extend(
            extract_fuel_table(
                workbook=workbook,
                sheet="Table_5",
                table_name="Table 5",
                group_name="jurisdiction",
            )
        )

    table1_latest = dedupe_latest(table1, ["vehicle_type", "year", "jurisdiction"])
    table3_latest = dedupe_latest(table3, ["vehicle_type", "year", "jurisdiction"])
    table4_latest = dedupe_latest(table4, ["vehicle_type", "year", "fuel_type"])
    table5_latest = dedupe_latest(table5, ["jurisdiction", "year", "fuel_type"])

    write_csv(
        OUT_DIR / "abs_vehicle_stock_by_type_all_releases.csv",
        table1,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "vehicles"],
    )
    write_csv(
        OUT_DIR / "abs_vehicle_stock_by_type_latest.csv",
        table1_latest,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "jurisdiction", "vehicles"],
    )
    write_csv(
        OUT_DIR / "abs_vehicle_average_age_by_type_all_releases.csv",
        table3,
        [
            "source_workbook",
            "release_year",
            "table",
            "vehicle_type",
            "year",
            "jurisdiction",
            "average_age_years",
        ],
    )
    write_csv(
        OUT_DIR / "abs_vehicle_average_age_by_type_latest.csv",
        table3_latest,
        [
            "source_workbook",
            "release_year",
            "table",
            "vehicle_type",
            "year",
            "jurisdiction",
            "average_age_years",
        ],
    )
    write_csv(
        OUT_DIR / "abs_fuel_by_vehicle_type_all_releases.csv",
        table4,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "fuel_type", "vehicles"],
    )
    write_csv(
        OUT_DIR / "abs_fuel_by_vehicle_type_latest.csv",
        table4_latest,
        ["source_workbook", "release_year", "table", "vehicle_type", "year", "fuel_type", "vehicles"],
    )
    write_csv(
        OUT_DIR / "abs_fuel_by_jurisdiction_all_releases.csv",
        table5,
        ["source_workbook", "release_year", "table", "jurisdiction", "year", "fuel_type", "vehicles"],
    )
    write_csv(
        OUT_DIR / "abs_fuel_by_jurisdiction_latest.csv",
        table5_latest,
        ["source_workbook", "release_year", "table", "jurisdiction", "year", "fuel_type", "vehicles"],
    )

    summary = [
        "# ABS Motor Vehicle Census Extraction Summary",
        "",
        f"Workbooks processed: {len(list(RAW_DIR.glob('*.xls')))}",
        f"Table 1 records, all releases: {len(table1)}",
        f"Table 3 records, all releases: {len(table3)}",
        f"Table 4 records, all releases: {len(table4)}",
        f"Table 5 records, all releases: {len(table5)}",
        "",
        f"Latest Table 5 records after overlap dedupe: {len(table5_latest)}",
        "",
        "Revision rule: overlapping year/category values keep the latest release year.",
        "Fuel caveat: ABS 'other' includes LPG, dual fuel and electric, so it is not EV stock.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
