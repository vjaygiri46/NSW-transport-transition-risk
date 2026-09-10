"""Generate final marginal uncertainty envelopes from selected benchmark models."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from marginal_benchmark import (  # noqa: E402
    FORECAST_YEARS,
    LOW_EMISSION_SERIES,
    METHODS,
    SeriesSpec,
    apply_bounds,
    apply_style,
    build_series,
    clean_xy,
    load_inputs,
    save_fig,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "marginal_final"
REPORT_DIR = ROOT / "reports"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bounded_short_series_envelope(
    spec: SeriesSpec,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    increments = np.diff(y)
    if len(increments) == 0:
        increments = np.array([0.0])
    conservative_step = max(0.0, float(np.quantile(increments, 0.25)) * 0.75)
    continuation_step = max(0.0, float(np.median(increments)))
    accelerated_step = max(continuation_step, float(np.quantile(increments, 0.75)) * 1.25)
    upper_bound = spec.upper_bound if spec.upper_bound is not None else 100.0

    p10 = p50 = p90 = float(y[-1])
    rows: list[dict[str, object]] = []
    for year in FORECAST_YEARS:
        p10 = min(upper_bound, p10 + conservative_step)
        p50 = min(upper_bound, p50 + continuation_step)
        p90 = min(upper_bound, p90 + accelerated_step)
        rows.append(
            {
                "series_id": spec.series_id,
                "label": spec.label,
                "year": year,
                "p10": p10,
                "p50": p50,
                "p90": max(p90, p50),
                "unit": spec.unit,
                "selected_method": "bounded_short_series_scenario",
                "uncertainty_method": "bounded scenario increments",
            }
        )

    summary = {
        "series_id": spec.series_id,
        "label": spec.label,
        "observed_start": int(x.min()),
        "observed_end": int(x.max()),
        "n_observations": len(y),
        "unit": spec.unit,
        "selected_method": "bounded_short_series_scenario",
        "uncertainty_method": "bounded scenario increments",
        "residual_q10": "",
        "residual_q90": "",
        "notes": "Scenario-constrained because clean low-emission detail has only five annual observations.",
    }
    return rows, summary


def rolling_errors(spec: SeriesSpec, x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    min_train = int(METHODS[method]["min_train"])
    errors: list[float] = []
    for origin in range(min_train, len(y)):
        train_x = x[:origin]
        train_y = y[:origin]
        target_year = np.array([x[origin]])
        try:
            pred = float(METHODS[method]["func"](train_x, train_y, target_year, spec)[0])
        except Exception:
            continue
        errors.append(pred - float(y[origin]))
    return np.array(errors, dtype=float)


def selected_method_for(spec: SeriesSpec) -> str:
    selected = {
        "fleet_total_nsw": "arima_drift",
        "petrol_share_nsw": "holt_linear",
        "diesel_share_nsw": "holt_linear",
        "renewable_share_nsw": "holt_linear",
        "road_energy_use_national": "arima_drift",
        "road_direct_emissions_national": "arima_drift",
        "sydney_passenger_activity": "arima_drift",
    }
    if spec.series_id in LOW_EMISSION_SERIES:
        return "bounded_short_series_scenario"
    return selected[spec.series_id]


def model_envelope(
    spec: SeriesSpec,
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target_years = np.array(FORECAST_YEARS, dtype=float)
    p50 = METHODS[method]["func"](x, y, target_years, spec)
    errors = rolling_errors(spec, x, y, method)
    if len(errors) >= 5:
        q10 = float(np.quantile(errors, 0.10))
        q90 = float(np.quantile(errors, 0.90))
    elif len(errors) > 0:
        scale = float(np.std(errors, ddof=1)) if len(errors) > 1 else abs(float(errors[0]))
        q10, q90 = -1.2816 * scale, 1.2816 * scale
    else:
        fitted = METHODS[method]["func"](x[:-1], y[:-1], np.array([x[-1]]), spec) if len(y) > 2 else np.array([y[-1]])
        scale = abs(float(fitted[0]) - float(y[-1]))
        q10, q90 = -1.2816 * scale, 1.2816 * scale

    rows: list[dict[str, object]] = []
    for idx, year in enumerate(FORECAST_YEARS, start=1):
        horizon_scale = float(np.sqrt(idx))
        lower = p50[idx - 1] + q10 * horizon_scale
        upper = p50[idx - 1] + q90 * horizon_scale
        p10, median, p90 = sorted([float(lower), float(p50[idx - 1]), float(upper)])
        if spec.monotonic == "decreasing":
            p90 = min(p90, float(y[-1]))
        elif spec.monotonic == "increasing":
            p10 = max(p10, float(y[-1]))
        if spec.lower_bound is not None:
            p10, median, p90 = max(p10, spec.lower_bound), max(median, spec.lower_bound), max(p90, spec.lower_bound)
        if spec.upper_bound is not None:
            p10, median, p90 = min(p10, spec.upper_bound), min(median, spec.upper_bound), min(p90, spec.upper_bound)
        p10, p90 = min(p10, median), max(p90, median)
        rows.append(
            {
                "series_id": spec.series_id,
                "label": spec.label,
                "year": year,
                "p10": p10,
                "p50": median,
                "p90": p90,
                "unit": spec.unit,
                "selected_method": method,
                "uncertainty_method": "rolling one-step residual quantiles with square-root horizon scaling",
            }
        )

    summary = {
        "series_id": spec.series_id,
        "label": spec.label,
        "observed_start": int(x.min()),
        "observed_end": int(x.max()),
        "n_observations": len(y),
        "unit": spec.unit,
        "selected_method": method,
        "uncertainty_method": "rolling one-step residual quantiles with square-root horizon scaling",
        "residual_q10": q10,
        "residual_q90": q90,
        "notes": "Selected from marginal benchmark; envelope is marginal only, not a joint simulation.",
    }
    return rows, summary


def plot_final_envelope(spec: SeriesSpec, x: np.ndarray, y: np.ndarray, rows: list[dict[str, object]]) -> None:
    forecast = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(x.astype(int), y.astype(float), color="tab:blue", marker="o", ms=2.5, label="Observed")
    years = forecast["year"].astype(int).to_numpy()
    p10 = forecast["p10"].astype(float).to_numpy()
    p50 = forecast["p50"].astype(float).to_numpy()
    p90 = forecast["p90"].astype(float).to_numpy()
    ax.fill_between(years, p10, p90, color="tab:green", alpha=0.22, linewidth=0, label="P10-P90")
    ax.plot(years, p50, color="tab:green", marker="s", ms=2.5, label="P50")
    ax.set_xlabel("Year")
    ax.set_ylabel(spec.unit)
    ax.set_title(spec.label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=6)
    ax.legend(loc="best", framealpha=0.85, handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    save_fig(fig, f"final_marginal_{spec.series_id}")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(summaries: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    scen = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    final_2030 = scen[scen["year"] == 2030].sort_values("series_id")
    lines = [
        "# Final Marginal Envelopes",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This report converts the selected marginal benchmark treatments into P10/P50/P90 envelopes for 2026-2030. These envelopes are the inputs for dependence modeling.",
        "",
        "## Method",
        "",
        "- Selected classical models come from `reports/marginal_model_benchmark.md`.",
        "- Classical-model uncertainty uses rolling one-step residual quantiles with square-root horizon scaling.",
        "- BEV/FCEV and hybrid shares retain bounded scenario envelopes because only five clean annual observations are available.",
        "- Bounds are enforced for shares and non-negative count, energy, emissions, and activity variables.",
        "",
        "## Selected Treatments",
        "",
    ]
    lines.extend(
        markdown_table(
            summary_df[["series_id", "selected_method", "uncertainty_method", "unit"]],
            ["series_id", "selected_method", "uncertainty_method", "unit"],
        )
    )
    lines.extend(["", "## 2030 Final Marginal Envelopes", ""])
    lines.extend(markdown_table(final_2030[["series_id", "p10", "p50", "p90", "unit"]], ["series_id", "p10", "p50", "p90", "unit"]))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `data_processed/marginal_final/final_marginal_envelopes_2026_2030.csv`",
            "- `data_processed/marginal_final/final_marginal_summary.csv`",
            "- `reports/figures/final_marginal_*.png` / `.pdf`",
            "",
            "## Use in the joint model",
            "",
            "Use these final marginal envelopes as the inputs to the dependence model. The earlier exploratory envelopes are not used in the final analysis.",
        ]
    )
    (REPORT_DIR / "final_marginal_envelopes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    data = load_inputs()
    series = build_series(data)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for spec, observed in series.values():
        x, y = clean_xy(spec, observed)
        method = selected_method_for(spec)
        if method == "bounded_short_series_scenario":
            rows, summary = bounded_short_series_envelope(spec, x, y)
        else:
            rows, summary = model_envelope(spec, x, y, method)
        all_rows.extend(rows)
        summaries.append(summary)
        plot_final_envelope(spec, x, y, rows)

    write_csv(
        OUT_DIR / "final_marginal_envelopes_2026_2030.csv",
        all_rows,
        ["series_id", "label", "year", "p10", "p50", "p90", "unit", "selected_method", "uncertainty_method"],
    )
    write_csv(
        OUT_DIR / "final_marginal_summary.csv",
        summaries,
        [
            "series_id",
            "label",
            "observed_start",
            "observed_end",
            "n_observations",
            "unit",
            "selected_method",
            "uncertainty_method",
            "residual_q10",
            "residual_q90",
            "notes",
        ],
    )
    write_report(summaries, all_rows)
    print(f"Wrote final marginal outputs to {OUT_DIR}")
    print(f"Wrote final marginal report to {REPORT_DIR / 'final_marginal_envelopes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
