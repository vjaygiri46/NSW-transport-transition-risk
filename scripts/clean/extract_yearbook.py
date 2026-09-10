"""Extract the BITRE Yearbook tables used in the analysis into tidy CSV files."""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path

from extract_road_vehicles import (
    clean_text,
    load_shared_strings,
    normalize_label,
    read_sheet,
    safe_number,
    sheet_paths,
    trim_leading_empty_columns,
)


ROOT = Path(__file__).resolve().parents[2]
YEARBOOK_DIR = ROOT / "data_raw" / "bitre" / "yearbook" / "bitre-yearbook-2025"
OUT_DIR = ROOT / "data_processed" / "bitre_yearbook"

ENERGY_WORKBOOK = YEARBOOK_DIR / "9. TRANSPORT ENERGY AND ENVIRONEMENT.xlsx"
PASSENGER_WORKBOOK = YEARBOOK_DIR / "2. PASSENGERS.xlsx"

ENERGY_TABLES = {
    "Table 9.3": {
        "topic": "transport_direct_ghg_by_mode",
        "unit": "Mt CO2-e",
        "measure": "direct_ghg_emissions",
    },
    "Table 9.4": {
        "topic": "road_direct_ghg_by_vehicle_type",
        "unit": "Mt CO2-e",
        "measure": "direct_ghg_emissions",
    },
    "Table 9.5": {
        "topic": "transport_full_fuel_cycle_ghg_by_mode",
        "unit": "Mt CO2-e",
        "measure": "full_fuel_cycle_ghg_emissions",
    },
    "Table 9.10": {
        "topic": "road_energy_use_by_vehicle_type",
        "unit": "PJ",
        "measure": "energy_use",
    },
    "Table 9.11": {
        "topic": "domestic_transport_energy_use_by_fuel_type",
        "unit": "PJ",
        "measure": "energy_use",
    },
}

PASSENGER_TABLES = {
    "Table 2.1": {
        "topic": "national_motorised_passenger_travel_by_mode",
        "unit": "billion passenger-km",
        "measure": "passenger_activity",
    }
}


def read_workbook(path: Path) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        paths = sheet_paths(zf)
        return {
            name: trim_leading_empty_columns(read_sheet(zf, sheet_path, shared))
            for name, sheet_path in paths.items()
        }


def find_financial_year_header(rows: list[list[str]], start: int = 0) -> int:
    for idx in range(start, len(rows)):
        if rows[idx] and normalize_label(rows[idx][0]) == "financial year":
            return idx
    raise ValueError("Could not find Financial year header")


def canonical_variable(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+[a-z]\.$", "", text)
    text = text.replace("***", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slug(value: str) -> str:
    text = canonical_variable(value).lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def is_period(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}", clean_text(value)))


def is_stop_row(row: list[str]) -> bool:
    nonempty = [clean_text(v) for v in row if clean_text(v)]
    if not nonempty:
        return False
    first = normalize_label(nonempty[0])
    return first.startswith("note") or first.startswith("source") or first.startswith("[")


def extract_wide_table(
    *,
    workbook: str,
    sheet_name: str,
    rows: list[list[str]],
    topic: str,
    measure: str,
    unit: str,
    section: str = "",
    start: int = 0,
    end: int | None = None,
) -> list[dict[str, object]]:
    table_rows = rows[:end] if end is not None else rows
    header_idx = find_financial_year_header(table_rows, start=start)
    header = table_rows[header_idx]
    variables = {
        idx: canonical_variable(value)
        for idx, value in enumerate(header)
        if idx > 0 and clean_text(value)
    }
    records: list[dict[str, object]] = []

    for row in table_rows[header_idx + 1 :]:
        if is_stop_row(row):
            break
        period = clean_text(row[0] if row else "")
        if not is_period(period):
            continue
        for idx, variable in variables.items():
            value = safe_number(row[idx] if idx < len(row) else "")
            if value is None:
                continue
            records.append(
                {
                    "source_workbook": workbook,
                    "sheet": sheet_name,
                    "topic": topic,
                    "section": section,
                    "period_type": "financial_year",
                    "period": period,
                    "variable": variable,
                    "variable_slug": slug(variable),
                    "measure": measure,
                    "value": value,
                    "unit": unit,
                }
            )
    return records


def find_stacked_sections(rows: list[list[str]], prefix: str) -> list[tuple[int, int, str]]:
    starts: list[tuple[int, str]] = []
    for idx, row in enumerate(rows):
        title = clean_text(row[0] if row else "")
        if title.startswith(prefix):
            section = title.split("—", 1)[-1].strip() if "—" in title else title
            starts.append((idx, section))
    sections: list[tuple[int, int, str]] = []
    for pos, (start, section) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(rows)
        sections.append((start, end, section))
    return sections


def write_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    energy_sheets = read_workbook(ENERGY_WORKBOOK)
    passenger_sheets = read_workbook(PASSENGER_WORKBOOK)

    energy_records: list[dict[str, object]] = []
    for sheet_name, meta in ENERGY_TABLES.items():
        energy_records.extend(
            extract_wide_table(
                workbook=ENERGY_WORKBOOK.name,
                sheet_name=sheet_name,
                rows=energy_sheets[sheet_name],
                topic=meta["topic"],
                measure=meta["measure"],
                unit=meta["unit"],
            )
        )

    passenger_records: list[dict[str, object]] = []
    for sheet_name, meta in PASSENGER_TABLES.items():
        passenger_records.extend(
            extract_wide_table(
                workbook=PASSENGER_WORKBOOK.name,
                sheet_name=sheet_name,
                rows=passenger_sheets[sheet_name],
                topic=meta["topic"],
                measure=meta["measure"],
                unit=meta["unit"],
                section="Australia",
            )
        )

    city_rows = passenger_sheets["Table 2.3a-i"]
    for start, end, city in find_stacked_sections(city_rows, "Table 2.3"):
        passenger_records.extend(
            extract_wide_table(
                workbook=PASSENGER_WORKBOOK.name,
                sheet_name="Table 2.3a-i",
                rows=city_rows,
                topic="capital_city_motorised_passenger_travel_by_mode",
                measure="passenger_activity",
                unit="billion passenger-km",
                section=city,
                start=start,
                end=end,
            )
        )

    fields = [
        "source_workbook",
        "sheet",
        "topic",
        "section",
        "period_type",
        "period",
        "variable",
        "variable_slug",
        "measure",
        "value",
        "unit",
    ]
    write_csv(OUT_DIR / "transport_energy_emissions_long.csv", energy_records, fields)
    write_csv(OUT_DIR / "passenger_activity_long.csv", passenger_records, fields)

    summary = [
        "# BITRE Yearbook Extraction Summary",
        "",
        f"Energy/emissions workbook: {ENERGY_WORKBOOK.name}",
        f"Passenger workbook: {PASSENGER_WORKBOOK.name}",
        f"Energy/emissions records: {len(energy_records)}",
        f"Passenger activity records: {len(passenger_records)}",
        "",
        "Primary outputs:",
        "- transport_energy_emissions_long.csv",
        "- passenger_activity_long.csv",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
