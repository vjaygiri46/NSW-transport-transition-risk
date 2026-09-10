"""Build harmonized model-input tables from cleaned source-specific outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data_processed"
OUT_DIR = PROCESSED / "model_inputs"

ABS_DIR = PROCESSED / "abs_motor_vehicle_census"
BITRE_ROAD_DIR = PROCESSED / "road_vehicles"
ENERGY_DIR = PROCESSED / "energy"
YEARBOOK_DIR = PROCESSED / "bitre_yearbook"

JURISDICTIONS = {"NSW", "AUS"}
LOW_EMISSION_TYPES = {
    "battery_fuel_cell_electric": "bev_fcev",
    "hybrid_electric": "hybrid",
    "dual_fuel": "dual_fuel",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def pct(numerator: float, denominator: float) -> float | str:
    if not denominator:
        return ""
    return numerator / denominator * 100.0


def add_yoy(rows: list[dict[str, object]], value_fields: list[str], group_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: int(row["year"]))
        previous: dict[str, float] = {}
        for row in group_rows:
            for field in value_fields:
                value = fnum(row.get(field))
                prior = previous.get(field)
                row[f"{field}_yoy_percent"] = pct(value - prior, prior) if prior not in {None, 0.0} else ""
                previous[field] = value
    return rows


def build_broad_fleet() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    abs_rows = read_csv(ABS_DIR / "abs_fuel_by_jurisdiction_latest.csv")
    bitre_rows = read_csv(BITRE_ROAD_DIR / "motive_power_by_jurisdiction_latest.csv")

    output: list[dict[str, object]] = []

    abs_grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in abs_rows:
        jurisdiction = row["jurisdiction"]
        if jurisdiction not in JURISDICTIONS:
            continue
        key = (jurisdiction, int(row["year"]))
        abs_grouped[key][row["fuel_type"]] = fnum(row["vehicles"])

    for (jurisdiction, year), values in sorted(abs_grouped.items()):
        petrol = values.get("petrol_total", 0.0)
        diesel = values.get("diesel", 0.0)
        transition_other = values.get("other", 0.0)
        total = values.get("total", 0.0)
        output.append(
            {
                "source_family": "ABS",
                "jurisdiction": jurisdiction,
                "year": year,
                "petrol_vehicles": petrol,
                "diesel_vehicles": diesel,
                "transition_or_other_vehicles": transition_other,
                "total_vehicles": total,
                "petrol_share_percent": pct(petrol, total),
                "diesel_share_percent": pct(diesel, total),
                "transition_or_other_share_percent": pct(transition_other, total),
                "definition_note": "ABS other includes LPG, dual fuel and electric; not EV-specific.",
            }
        )

    bitre_grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in bitre_rows:
        jurisdiction = row["jurisdiction"]
        if jurisdiction not in JURISDICTIONS:
            continue
        key = (jurisdiction, int(row["year"]))
        bitre_grouped[key][row["motive_power"]] = fnum(row["vehicles"])

    for (jurisdiction, year), values in sorted(bitre_grouped.items()):
        petrol = values.get("petrol", 0.0)
        diesel = values.get("diesel", 0.0)
        transition_other = (
            values.get("dual_fuel", 0.0)
            + values.get("hybrid_electric", 0.0)
            + values.get("battery_fuel_cell_electric", 0.0)
            + values.get("other", 0.0)
            + values.get("other_or_not_specified", 0.0)
        )
        total = values.get("total", 0.0)
        output.append(
            {
                "source_family": "BITRE",
                "jurisdiction": jurisdiction,
                "year": year,
                "petrol_vehicles": petrol,
                "diesel_vehicles": diesel,
                "transition_or_other_vehicles": transition_other,
                "total_vehicles": total,
                "petrol_share_percent": pct(petrol, total),
                "diesel_share_percent": pct(diesel, total),
                "transition_or_other_share_percent": pct(transition_other, total),
                "definition_note": "BITRE transition/other sums dual fuel, hybrid, BEV/FCEV and residual other categories.",
            }
        )

    preferred = [
        row
        for row in output
        if (row["source_family"] == "ABS" and int(row["year"]) <= 2020)
        or (row["source_family"] == "BITRE" and int(row["year"]) >= 2021)
    ]
    add_yoy(preferred, ["total_vehicles", "petrol_vehicles", "diesel_vehicles"], ["jurisdiction"])
    return output, preferred


def build_low_emission_detail() -> list[dict[str, object]]:
    bitre_rows = read_csv(BITRE_ROAD_DIR / "motive_power_by_jurisdiction_latest.csv")
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in bitre_rows:
        jurisdiction = row["jurisdiction"]
        if jurisdiction not in JURISDICTIONS:
            continue
        grouped[(jurisdiction, int(row["year"]))][row["motive_power"]] = fnum(row["vehicles"])

    output: list[dict[str, object]] = []
    for (jurisdiction, year), values in sorted(grouped.items()):
        total = values.get("total", 0.0)
        bev = values.get("battery_fuel_cell_electric", 0.0)
        hybrid = values.get("hybrid_electric", 0.0)
        dual = values.get("dual_fuel", 0.0)
        low_emission = bev + hybrid
        output.append(
            {
                "jurisdiction": jurisdiction,
                "year": year,
                "bev_fcev_vehicles": bev,
                "hybrid_vehicles": hybrid,
                "dual_fuel_vehicles": dual,
                "low_emission_vehicles_bev_hybrid": low_emission,
                "total_vehicles": total,
                "bev_fcev_share_percent": pct(bev, total),
                "hybrid_share_percent": pct(hybrid, total),
                "low_emission_share_percent": pct(low_emission, total),
            }
        )
    add_yoy(output, ["bev_fcev_vehicles", "hybrid_vehicles", "low_emission_vehicles_bev_hybrid"], ["jurisdiction"])
    return output


def build_electricity_mix() -> list[dict[str, object]]:
    rows = read_csv(ENERGY_DIR / "table_o_mix_metrics.csv")
    output = []
    for row in rows:
        if row["geography"] not in JURISDICTIONS:
            continue
        output.append(
            {
                "jurisdiction": row["geography"],
                "period_type": row["period_type"],
                "period": row["period"],
                "is_estimate": row["is_estimate"],
                "total_generation_gwh": row["total_generation_gwh"],
                "renewable_generation_gwh": row["renewable_generation_gwh"],
                "non_renewable_generation_gwh": row["non_renewable_generation_gwh"],
                "renewable_share_percent": row["renewable_share_percent"],
                "non_renewable_share_percent": row["non_renewable_share_percent"],
            }
        )
    return output


def build_transport_calibration() -> list[dict[str, object]]:
    rows = read_csv(YEARBOOK_DIR / "transport_energy_emissions_long.csv")
    wanted = {
        ("road_direct_ghg_by_vehicle_type", "total_road"),
        ("road_energy_use_by_vehicle_type", "total_road"),
        ("domestic_transport_energy_use_by_fuel_type", "electricity"),
        ("transport_full_fuel_cycle_ghg_by_mode", "road_vehicles"),
    }
    output = []
    for row in rows:
        if (row["topic"], row["variable_slug"]) not in wanted:
            continue
        output.append(
            {
                "period": row["period"],
                "topic": row["topic"],
                "variable": row["variable"],
                "measure": row["measure"],
                "value": row["value"],
                "unit": row["unit"],
            }
        )
    return output


def build_sydney_activity() -> list[dict[str, object]]:
    rows = read_csv(YEARBOOK_DIR / "passenger_activity_long.csv")
    output = []
    for row in rows:
        if row["topic"] == "capital_city_motorised_passenger_travel_by_mode" and row["section"] == "Sydney":
            if row["variable_slug"] in {"total", "passenger_cars", "commercial_vehicles", "bus", "heavy_rail", "light_rail"}:
                output.append(
                    {
                        "period": row["period"],
                        "mode": row["variable"],
                        "mode_slug": row["variable_slug"],
                        "value": row["value"],
                        "unit": row["unit"],
                    }
                )
    return output


def main() -> int:
    broad_all, broad_preferred = build_broad_fleet()
    low_emission = build_low_emission_detail()
    electricity = build_electricity_mix()
    transport = build_transport_calibration()
    sydney_activity = build_sydney_activity()

    write_csv(
        OUT_DIR / "fleet_broad_all_sources.csv",
        broad_all,
        [
            "source_family",
            "jurisdiction",
            "year",
            "petrol_vehicles",
            "diesel_vehicles",
            "transition_or_other_vehicles",
            "total_vehicles",
            "petrol_share_percent",
            "diesel_share_percent",
            "transition_or_other_share_percent",
            "definition_note",
        ],
    )
    write_csv(
        OUT_DIR / "fleet_broad_preferred_2010_2025.csv",
        broad_preferred,
        [
            "source_family",
            "jurisdiction",
            "year",
            "petrol_vehicles",
            "diesel_vehicles",
            "transition_or_other_vehicles",
            "total_vehicles",
            "petrol_share_percent",
            "diesel_share_percent",
            "transition_or_other_share_percent",
            "total_vehicles_yoy_percent",
            "petrol_vehicles_yoy_percent",
            "diesel_vehicles_yoy_percent",
            "definition_note",
        ],
    )
    write_csv(
        OUT_DIR / "low_emission_detail_2021_2025.csv",
        low_emission,
        [
            "jurisdiction",
            "year",
            "bev_fcev_vehicles",
            "hybrid_vehicles",
            "dual_fuel_vehicles",
            "low_emission_vehicles_bev_hybrid",
            "total_vehicles",
            "bev_fcev_share_percent",
            "hybrid_share_percent",
            "low_emission_share_percent",
            "bev_fcev_vehicles_yoy_percent",
            "hybrid_vehicles_yoy_percent",
            "low_emission_vehicles_bev_hybrid_yoy_percent",
        ],
    )
    write_csv(
        OUT_DIR / "electricity_mix_nsw_aus.csv",
        electricity,
        [
            "jurisdiction",
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
    write_csv(
        OUT_DIR / "transport_calibration_national.csv",
        transport,
        ["period", "topic", "variable", "measure", "value", "unit"],
    )
    write_csv(
        OUT_DIR / "sydney_activity_proxy.csv",
        sydney_activity,
        ["period", "mode", "mode_slug", "value", "unit"],
    )

    summary = [
        "# Model Input Build Summary",
        "",
        f"Broad fleet rows, all sources: {len(broad_all)}",
        f"Broad fleet rows, preferred bridge: {len(broad_preferred)}",
        f"Low-emission detail rows: {len(low_emission)}",
        f"Electricity mix rows: {len(electricity)}",
        f"Transport calibration rows: {len(transport)}",
        f"Sydney activity rows: {len(sydney_activity)}",
        "",
        "Bridge rule:",
        "- ABS broad fuel/fleet series is used for 2010-2020.",
        "- BITRE broad and detailed motive-power series is used for 2021-2025.",
        "- The 2021 ABS/BITRE overlap remains available in `fleet_broad_all_sources.csv` for discontinuity checks.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
