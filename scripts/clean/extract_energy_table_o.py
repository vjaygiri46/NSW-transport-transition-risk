"""Extract Australian Energy Statistics Table O electricity data.

The project environment does not currently have openpyxl, so this uses the
same lightweight XLSX XML reader as the road-vehicle extractor.
"""

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
RAW_DIR = ROOT / "data_raw" / "energy" / "table_o"
OUT_DIR = ROOT / "data_processed" / "energy"
PRIMARY_WORKBOOK = RAW_DIR / "table-o-electricity-generation-by-fuel type-2024-25-and-2025.xlsx"

SERIES_SHEETS = {
    "AUS FY": ("AUS", "financial_year"),
    "NSW FY": ("NSW", "financial_year"),
    "AUS CY": ("AUS", "calendar_year"),
    "NSW CY": ("NSW", "calendar_year"),
}

SUMMARY_SHEETS = {
    "State summary 2024-25": ("2024-25", "financial_year"),
    "State summary 2025": ("2025", "calendar_year"),
}

JURISDICTIONS = {"NSW", "VIC", "QLD", "WA", "SA", "TAS", "NT", "AUS"}

FUEL_GROUPS = {
    "black coal": "non_renewable",
    "brown coal": "non_renewable",
    "natural gas": "non_renewable",
    "oil products": "non_renewable",
    "other [note a]": "non_renewable",
    "total non-renewable": "subtotal_non_renewable",
    "biomass": "renewable",
    "bagasse, wood": "renewable",
    "biogas": "renewable",
    "wind": "renewable",
    "hydro": "renewable",
    "large-scale solar pv": "renewable",
    "small-scale solar pv": "renewable",
    "geothermal": "renewable",
    "total renewable": "subtotal_renewable",
    "total": "total",
    "per cent renewable generation": "metric_percent",
}


def canonical_fuel(value: str) -> str:
    label = normalize_label(value)
    label = label.replace("non renewable", "non-renewable")
    return label


def find_header_row(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows):
        if row and normalize_label(row[0]) == "fuel type":
            return idx
    raise ValueError("Could not find 'Fuel type' header row")


def clean_period(value: str) -> tuple[str, bool]:
    text = clean_text(value)
    is_estimate = "(est" in text.lower()
    text = re.sub(r"\s*\(est\.?\)\s*", "", text, flags=re.IGNORECASE)
    return text, is_estimate


def is_data_label(label: str) -> bool:
    if not label or label.startswith("["):
        return False
    if label.startswith("of which"):
        return False
    return label in FUEL_GROUPS


def read_primary_workbook() -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(PRIMARY_WORKBOOK) as zf:
        shared = load_shared_strings(zf)
        paths = sheet_paths(zf)
        return {
            name: trim_leading_empty_columns(read_sheet(zf, path, shared))
            for name, path in paths.items()
        }


def extract_series_sheet(
    *, source_workbook: str, sheet_name: str, geography: str, period_type: str, rows: list[list[str]]
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows)
    header = rows[header_idx]
    periods = {idx: clean_period(value) for idx, value in enumerate(header) if idx > 0 and clean_text(value)}
    records: list[dict[str, object]] = []

    for row in rows[header_idx + 1 :]:
        if not row:
            continue
        fuel_type = canonical_fuel(row[0])
        if not is_data_label(fuel_type):
            continue
        for idx, (period, is_estimate) in periods.items():
            value = safe_number(row[idx] if idx < len(row) else "")
            if value is None:
                continue
            records.append(
                {
                    "source_workbook": source_workbook,
                    "sheet": sheet_name,
                    "geography": geography,
                    "period_type": period_type,
                    "period": period,
                    "fuel_type": fuel_type,
                    "fuel_group": FUEL_GROUPS[fuel_type],
                    "generation_gwh": value,
                    "is_estimate": str(is_estimate).lower(),
                }
            )
    return records


def extract_summary_sheet(
    *, source_workbook: str, sheet_name: str, period: str, period_type: str, rows: list[list[str]]
) -> list[dict[str, object]]:
    header_idx = find_header_row(rows)
    header = rows[header_idx]
    jurisdictions = {
        idx: clean_text(value)
        for idx, value in enumerate(header)
        if idx > 0 and clean_text(value) in JURISDICTIONS
    }
    records: list[dict[str, object]] = []

    for row in rows[header_idx + 1 :]:
        if not row:
            continue
        fuel_type = canonical_fuel(row[0])
        if not is_data_label(fuel_type):
            continue
        is_percent = FUEL_GROUPS[fuel_type] == "metric_percent"
        for idx, geography in jurisdictions.items():
            value = safe_number(row[idx] if idx < len(row) else "")
            if value is None:
                continue
            records.append(
                {
                    "source_workbook": source_workbook,
                    "sheet": sheet_name,
                    "geography": geography,
                    "period_type": period_type,
                    "period": period,
                    "fuel_type": fuel_type,
                    "fuel_group": FUEL_GROUPS[fuel_type],
                    "value": value,
                    "unit": "percent" if is_percent else "GWh",
                    "is_estimate": str(period == "2025").lower(),
                }
            )
    return records


def build_mix_metrics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[object, ...], dict[str, float]] = {}
    meta: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        key = (record["geography"], record["period_type"], record["period"])
        meta[key] = {
            "geography": record["geography"],
            "period_type": record["period_type"],
            "period": record["period"],
            "is_estimate": record["is_estimate"],
        }
        if record["fuel_type"] in {"total renewable", "total non-renewable", "total"}:
            by_key.setdefault(key, {})[str(record["fuel_type"])] = float(record["generation_gwh"])

    output: list[dict[str, object]] = []
    for key, values in sorted(by_key.items(), key=lambda item: tuple(str(v) for v in item[0])):
        total = values.get("total")
        renewable = values.get("total renewable")
        non_renewable = values.get("total non-renewable")
        if not total:
            continue
        row = dict(meta[key])
        row.update(
            {
                "total_generation_gwh": total,
                "renewable_generation_gwh": renewable if renewable is not None else "",
                "non_renewable_generation_gwh": non_renewable if non_renewable is not None else "",
                "renewable_share_percent": (renewable / total * 100.0) if renewable is not None else "",
                "non_renewable_share_percent": (non_renewable / total * 100.0) if non_renewable is not None else "",
            }
        )
        output.append(row)
    return output


def write_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    sheets = read_primary_workbook()
    source_workbook = PRIMARY_WORKBOOK.name

    series_records: list[dict[str, object]] = []
    for sheet_name, (geography, period_type) in SERIES_SHEETS.items():
        series_records.extend(
            extract_series_sheet(
                source_workbook=source_workbook,
                sheet_name=sheet_name,
                geography=geography,
                period_type=period_type,
                rows=sheets[sheet_name],
            )
        )

    summary_records: list[dict[str, object]] = []
    for sheet_name, (period, period_type) in SUMMARY_SHEETS.items():
        summary_records.extend(
            extract_summary_sheet(
                source_workbook=source_workbook,
                sheet_name=sheet_name,
                period=period,
                period_type=period_type,
                rows=sheets[sheet_name],
            )
        )

    mix_metrics = build_mix_metrics(series_records)

    write_csv(
        OUT_DIR / "table_o_generation_by_fuel_long.csv",
        series_records,
        [
            "source_workbook",
            "sheet",
            "geography",
            "period_type",
            "period",
            "fuel_type",
            "fuel_group",
            "generation_gwh",
            "is_estimate",
        ],
    )
    write_csv(
        OUT_DIR / "table_o_state_summary_long.csv",
        summary_records,
        [
            "source_workbook",
            "sheet",
            "geography",
            "period_type",
            "period",
            "fuel_type",
            "fuel_group",
            "value",
            "unit",
            "is_estimate",
        ],
    )
    write_csv(
        OUT_DIR / "table_o_mix_metrics.csv",
        mix_metrics,
        [
            "geography",
            "period_type",
            "period",
            "is_estimate",
            "total_generation_gwh",
            "renewable_generation_gwh",
            "non_renewable_generation_gwh",
            "renewable_share_percent",
            "non_renewable_share_percent",
        ],
    )

    summary = [
        "# Energy Table O Extraction Summary",
        "",
        f"Workbook processed: {source_workbook}",
        f"Series records: {len(series_records)}",
        f"State-summary records: {len(summary_records)}",
        f"Mix metric records: {len(mix_metrics)}",
        "",
        "Primary outputs:",
        "- table_o_generation_by_fuel_long.csv",
        "- table_o_state_summary_long.csv",
        "- table_o_mix_metrics.csv",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
