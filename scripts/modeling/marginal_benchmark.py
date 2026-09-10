"""Benchmark classical marginal forecasting options for the model variables."""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data_processed" / "model_inputs"
OUT_DIR = ROOT / "data_processed" / "marginal_benchmark"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

FORECAST_YEARS = list(range(2026, 2031))
LOW_EMISSION_SERIES = {"bev_fcev_share_nsw", "hybrid_share_nsw"}


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    label: str
    unit: str
    year_col: str
    value_col: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    monotonic: str | None = None


def apply_style() -> None:
    if scienceplots is not None:
        try:
            plt.style.use(["science", "no-latex"])
        except Exception:
            plt.style.use("default")
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (3.5, 2.625),
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.0,
            "lines.linewidth": 1.0,
            "text.usetex": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "ytick.major.size": 3,
            "xtick.major.size": 3,
            "legend.frameon": True,
            "savefig.dpi": 600,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "fleet": pd.read_csv(INPUT_DIR / "fleet_broad_preferred_2010_2025.csv"),
        "low": pd.read_csv(INPUT_DIR / "low_emission_detail_2021_2025.csv"),
        "electricity": pd.read_csv(INPUT_DIR / "electricity_mix_nsw_aus.csv"),
        "transport": pd.read_csv(INPUT_DIR / "transport_calibration_national.csv"),
        "sydney": pd.read_csv(INPUT_DIR / "sydney_activity_proxy.csv"),
    }


def road_topic_series(transport: pd.DataFrame, topic: str, variable: str) -> pd.DataFrame:
    sub = transport[(transport["topic"] == topic) & (transport["variable"] == variable)].copy()
    sub["year"] = sub["period"].str.slice(0, 4).astype(int)
    sub["value"] = sub["value"].astype(float)
    return sub[["year", "value"]]


def build_series(data: dict[str, pd.DataFrame]) -> dict[str, tuple[SeriesSpec, pd.DataFrame]]:
    fleet = data["fleet"][data["fleet"]["jurisdiction"] == "NSW"].copy()
    low = data["low"][data["low"]["jurisdiction"] == "NSW"].copy()
    electricity = data["electricity"][
        (data["electricity"]["jurisdiction"] == "NSW")
        & (data["electricity"]["period_type"] == "calendar_year")
    ].copy()
    electricity["year"] = electricity["period"].astype(int)
    sydney = data["sydney"][data["sydney"]["mode_slug"] == "total"].copy()
    sydney["year"] = sydney["period"].str.slice(0, 4).astype(int)

    series: dict[str, tuple[SeriesSpec, pd.DataFrame]] = {
        "fleet_total_nsw": (
            SeriesSpec("fleet_total_nsw", "NSW total registered vehicles", "vehicles", "year", "total_vehicles", 0.0),
            fleet,
        ),
        "petrol_share_nsw": (
            SeriesSpec(
                "petrol_share_nsw",
                "NSW petrol share",
                "percent",
                "year",
                "petrol_share_percent",
                0.0,
                100.0,
                "decreasing",
            ),
            fleet,
        ),
        "diesel_share_nsw": (
            SeriesSpec("diesel_share_nsw", "NSW diesel share", "percent", "year", "diesel_share_percent", 0.0, 100.0),
            fleet,
        ),
        "bev_fcev_share_nsw": (
            SeriesSpec(
                "bev_fcev_share_nsw",
                "NSW BEV/FCEV share",
                "percent",
                "year",
                "bev_fcev_share_percent",
                0.0,
                100.0,
                "increasing",
            ),
            low,
        ),
        "hybrid_share_nsw": (
            SeriesSpec(
                "hybrid_share_nsw",
                "NSW hybrid share",
                "percent",
                "year",
                "hybrid_share_percent",
                0.0,
                100.0,
                "increasing",
            ),
            low,
        ),
        "renewable_share_nsw": (
            SeriesSpec(
                "renewable_share_nsw",
                "NSW renewable electricity share",
                "percent",
                "year",
                "renewable_share_percent",
                0.0,
                100.0,
                "increasing",
            ),
            electricity,
        ),
        "road_energy_use_national": (
            SeriesSpec("road_energy_use_national", "National road energy use", "PJ", "year", "value", 0.0),
            road_topic_series(data["transport"], "road_energy_use_by_vehicle_type", "Total road"),
        ),
        "road_direct_emissions_national": (
            SeriesSpec(
                "road_direct_emissions_national",
                "National road direct emissions",
                "Mt CO2-e",
                "year",
                "value",
                0.0,
            ),
            road_topic_series(data["transport"], "road_direct_ghg_by_vehicle_type", "Total road"),
        ),
        "sydney_passenger_activity": (
            SeriesSpec(
                "sydney_passenger_activity",
                "Sydney passenger activity",
                "billion passenger-km",
                "year",
                "value",
                0.0,
            ),
            sydney,
        ),
    }
    return series


def clean_xy(spec: SeriesSpec, observed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    obs = observed[[spec.year_col, spec.value_col]].dropna().copy()
    obs[spec.year_col] = obs[spec.year_col].astype(int)
    obs[spec.value_col] = obs[spec.value_col].astype(float)
    obs = obs.sort_values(spec.year_col)
    return obs[spec.year_col].to_numpy(dtype=float), obs[spec.value_col].to_numpy(dtype=float)


def apply_bounds(values: np.ndarray, spec: SeriesSpec, last_value: float | None = None) -> np.ndarray:
    bounded = values.astype(float).copy()
    if spec.monotonic == "decreasing" and last_value is not None:
        bounded = np.minimum.accumulate(np.r_[last_value, bounded])[1:]
    elif spec.monotonic == "increasing" and last_value is not None:
        bounded = np.maximum.accumulate(np.r_[last_value, bounded])[1:]
    if spec.lower_bound is not None:
        bounded = np.maximum(bounded, spec.lower_bound)
    if spec.upper_bound is not None:
        bounded = np.minimum(bounded, spec.upper_bound)
    return bounded


def predict_trend(x: np.ndarray, y: np.ndarray, target_years: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    slope, intercept = np.polyfit(x, y, 1)
    return apply_bounds(intercept + slope * target_years, spec, y[-1])


def predict_drift(x: np.ndarray, y: np.ndarray, target_years: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    if len(y) < 2:
        forecast = np.repeat(y[-1], len(target_years))
    else:
        yearly_step = float(np.mean(np.diff(y) / np.diff(x)))
        forecast = y[-1] + yearly_step * (target_years - x[-1])
    return apply_bounds(forecast, spec, y[-1])


def holt_sse(params: np.ndarray, y: np.ndarray) -> float:
    alpha, beta = params
    level = float(y[0])
    trend = float(y[1] - y[0]) if len(y) > 1 else 0.0
    sse = 0.0
    for value in y[1:]:
        forecast = level + trend
        sse += float((value - forecast) ** 2)
        new_level = alpha * value + (1.0 - alpha) * forecast
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
    return sse


def predict_holt(x: np.ndarray, y: np.ndarray, target_years: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    if len(y) < 4:
        return predict_drift(x, y, target_years, spec)
    result = minimize(
        holt_sse,
        x0=np.array([0.45, 0.15]),
        args=(y,),
        bounds=[(0.001, 0.999), (0.001, 0.999)],
        method="L-BFGS-B",
    )
    alpha, beta = result.x if result.success else (0.45, 0.15)
    level = float(y[0])
    trend = float(y[1] - y[0])
    for value in y[1:]:
        forecast = level + trend
        new_level = alpha * value + (1.0 - alpha) * forecast
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
    horizon = target_years - x[-1]
    return apply_bounds(level + horizon * trend, spec, y[-1])


def predict_arima_110(x: np.ndarray, y: np.ndarray, target_years: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    diffs = np.diff(y)
    if len(diffs) < 4:
        return predict_drift(x, y, target_years, spec)
    response = diffs[1:]
    lag = diffs[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    intercept, phi = np.linalg.lstsq(design, response, rcond=None)[0]
    phi = float(np.clip(phi, -0.95, 0.95))
    last_diff = float(diffs[-1])
    values: list[float] = []
    current = float(y[-1])
    for _ in target_years:
        next_diff = float(intercept + phi * last_diff)
        current += next_diff
        values.append(current)
        last_diff = next_diff
    return apply_bounds(np.array(values), spec, y[-1])


def logistic_curve(t: np.ndarray, cap: float, rate: float, midpoint: float) -> np.ndarray:
    return cap / (1.0 + np.exp(-rate * (t - midpoint)))


def predict_logistic(x: np.ndarray, y: np.ndarray, target_years: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    if spec.upper_bound is None or len(y) < 5:
        raise ValueError("constrained logistic requires at least five bounded observations")
    cap_upper = min(spec.upper_bound, max(100.0, float(np.max(y)) * 10.0))
    lower_cap = max(float(np.max(y)) * 1.05, 0.01)
    p0 = [min(cap_upper, max(20.0, float(np.max(y)) * 3.0)), 0.35, float(np.median(x) + 8.0)]
    bounds = ([lower_cap, 0.001, float(np.min(x) - 30.0)], [cap_upper, 3.0, float(np.max(x) + 50.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(logistic_curve, x, y, p0=p0, bounds=bounds, maxfev=20000)
    return apply_bounds(logistic_curve(target_years, *params), spec, y[-1])


METHODS = {
    "linear_trend": {"func": predict_trend, "min_train": 5, "class": "trend baseline"},
    "holt_linear": {"func": predict_holt, "min_train": 6, "class": "ETS/Holt linear"},
    "arima_drift": {"func": predict_drift, "min_train": 6, "class": "ARIMA(0,1,0) with drift"},
    "arima_110": {"func": predict_arima_110, "min_train": 9, "class": "ARIMA(1,1,0)-style differenced AR(1)"},
    "constrained_logistic": {"func": predict_logistic, "min_train": 5, "class": "bounded adoption sensitivity"},
}


def available_methods(spec: SeriesSpec, n: int) -> list[str]:
    methods = ["linear_trend", "holt_linear", "arima_drift", "arima_110"]
    if spec.upper_bound is not None and spec.monotonic == "increasing":
        methods.append("constrained_logistic")
    return [method for method in methods if n >= METHODS[method]["min_train"]]


def benchmark_one_step(spec: SeriesSpec, x: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = available_methods(spec, len(y))
    for method in methods:
        min_train = int(METHODS[method]["min_train"])
        errors: list[float] = []
        abs_pct_errors: list[float] = []
        violations = 0
        origins = range(min_train, len(y))
        for origin in origins:
            train_x = x[:origin]
            train_y = y[:origin]
            target_year = np.array([x[origin]])
            try:
                pred = float(METHODS[method]["func"](train_x, train_y, target_year, spec)[0])
            except Exception:
                continue
            actual = float(y[origin])
            if spec.lower_bound is not None and pred < spec.lower_bound - 1e-9:
                violations += 1
            if spec.upper_bound is not None and pred > spec.upper_bound + 1e-9:
                violations += 1
            errors.append(pred - actual)
            if abs(actual) > 1e-12:
                abs_pct_errors.append(abs((pred - actual) / actual) * 100.0)
        if not errors:
            continue
        err = np.array(errors, dtype=float)
        rows.append(
            {
                "series_id": spec.series_id,
                "method": method,
                "method_class": METHODS[method]["class"],
                "n_observations": len(y),
                "n_backtests": len(errors),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mape_percent": float(np.mean(abs_pct_errors)) if abs_pct_errors else "",
                "bias": float(np.mean(err)),
                "constraint_violations": violations,
            }
        )
    return rows


def forecast_methods(spec: SeriesSpec, x: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_years = np.array(FORECAST_YEARS, dtype=float)
    for method in available_methods(spec, len(y)):
        try:
            forecast = METHODS[method]["func"](x, y, target_years, spec)
        except Exception as exc:
            rows.append(
                {
                    "series_id": spec.series_id,
                    "method": method,
                    "year": "",
                    "forecast": "",
                    "unit": spec.unit,
                    "status": f"failed: {exc}",
                }
            )
            continue
        for year, value in zip(FORECAST_YEARS, forecast):
            rows.append(
                {
                    "series_id": spec.series_id,
                    "method": method,
                    "year": year,
                    "forecast": float(value),
                    "unit": spec.unit,
                    "status": "ok",
                }
            )
    return rows


def choose_method(spec: SeriesSpec, metrics: pd.DataFrame) -> dict[str, object]:
    if spec.series_id in LOW_EMISSION_SERIES:
        return {
            "series_id": spec.series_id,
            "selected_treatment": "bounded scenario envelope",
            "selected_method": "bounded_short_series_scenario",
            "reason": "Clean low-emission detail has only five annual observations; fitted adoption curves remain sensitivity checks.",
        }
    if metrics.empty:
        return {
            "series_id": spec.series_id,
            "selected_treatment": "trend baseline",
            "selected_method": "linear_trend",
            "reason": "Feasibility: no candidate passed the backtest check, so the transparent trend baseline was retained.",
        }
    valid = metrics[metrics["constraint_violations"].astype(float) == 0].copy()
    if valid.empty:
        valid = metrics.copy()
    enough_origins = valid[valid["n_backtests"].astype(int) >= 5].copy()
    if not enough_origins.empty:
        valid = enough_origins
    valid = valid.sort_values(["rmse", "mae"])
    best = valid.iloc[0]
    simple = valid[valid["method"] == "linear_trend"]
    if not simple.empty and float(simple.iloc[0]["rmse"]) <= 1.05 * float(best["rmse"]):
        chosen = simple.iloc[0]
        reason = "Parsimony: the linear trend was retained because its out-of-sample RMSE is within 5 percent of the best feasible model."
    else:
        chosen = best
        reason = "Out-of-sample accuracy: the lowest rolling one-step RMSE was selected among feasible classical methods with at least five backtest origins where available."
    return {
        "series_id": spec.series_id,
        "selected_treatment": str(chosen["method_class"]),
        "selected_method": str(chosen["method"]),
        "reason": reason,
    }


def plot_forecast_comparison(
    spec: SeriesSpec,
    x: np.ndarray,
    y: np.ndarray,
    forecasts: pd.DataFrame,
    selected_method: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(x.astype(int), y, color="black", marker="o", ms=2.4, label="Observed")
    sub = forecasts[(forecasts["series_id"] == spec.series_id) & (forecasts["status"] == "ok")].copy()
    methods = [m for m in sub["method"].dropna().unique() if m != "constrained_logistic"]
    if "constrained_logistic" in sub["method"].dropna().unique():
        methods.append("constrained_logistic")
    for method in methods:
        method_rows = sub[sub["method"] == method]
        width = 1.2 if method == selected_method else 0.8
        alpha = 0.95 if method == selected_method else 0.55
        ax.plot(
            method_rows["year"].astype(int),
            method_rows["forecast"].astype(float),
            marker="s",
            ms=2.2,
            linewidth=width,
            alpha=alpha,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("Year")
    ax.set_ylabel(spec.unit)
    ax.set_title(spec.label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=6)
    ax.legend(loc="best", framealpha=0.85, handlelength=1.3, borderpad=0.25, labelspacing=0.22)
    fig.tight_layout()
    save_fig(fig, f"marginal_benchmark_{spec.series_id}")


def plot_mape_summary(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    usable = metrics[metrics["mape_percent"] != ""].copy()
    if usable.empty:
        return
    usable["mape_percent"] = usable["mape_percent"].astype(float)
    best = usable.sort_values(["series_id", "mape_percent"]).groupby("series_id", as_index=False).first()
    labels = [s.replace("_", "\n") for s in best["series_id"]]
    fig, ax = plt.subplots(figsize=(7.0, 2.625))
    ax.bar(np.arange(len(best)), best["mape_percent"].astype(float), color="tab:blue", alpha=0.8)
    ax.set_xticks(np.arange(len(best)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=5.2)
    ax.set_ylabel("Best one-step MAPE (%)")
    ax.set_title("Marginal benchmark: best rolling one-step MAPE")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "marginal_benchmark_mape_summary")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    series_meta: list[dict[str, object]],
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Marginal Model Benchmark",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This benchmark tests classical marginal forecasting options before dependence modeling. It does not perform joint simulation.",
        "",
        "## Candidate Methods",
        "",
        "- Linear trend baseline.",
        "- Holt linear exponential smoothing implemented directly.",
        "- Random-walk-with-drift, equivalent to an ARIMA(0,1,0)-with-drift baseline.",
        "- Differenced AR(1), used as an ARIMA(1,1,0)-style candidate.",
        "- Constrained logistic sensitivity for bounded increasing shares only.",
        "",
        "No machine-learning model is used. The available data are annual and small-sample, so interpretability and constraint behavior matter as much as point error.",
        "",
        "## Series Coverage",
        "",
    ]
    meta_df = pd.DataFrame(series_meta)
    lines.extend(markdown_table(meta_df, ["series_id", "observed_years", "n_observations", "unit", "benchmark_status"]))
    lines.extend(["", "## Selected Marginal Treatments", ""])
    lines.extend(markdown_table(selected, ["series_id", "selected_method", "selected_treatment", "reason"]))
    lines.extend(["", "## Backtest Metrics", ""])
    if metrics.empty:
        lines.append("No rolling backtests were feasible.")
    else:
        metric_view = metrics.sort_values(["series_id", "rmse"])[
            ["series_id", "method", "n_backtests", "mae", "rmse", "mape_percent", "bias", "constraint_violations"]
        ].copy()
        lines.extend(markdown_table(metric_view, list(metric_view.columns)))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current marginal envelopes remain the baseline for comparison.",
            "- Longer annual series can be compared with classical alternatives using rolling one-step errors.",
            "- BEV/FCEV and hybrid shares remain scenario-constrained because five annual observations are not enough to select a fitted adoption curve with confidence.",
            "- Dependence modeling should use the selected treatments above, not raw benchmark candidates.",
            "",
            "## Outputs",
            "",
            "- `data_processed/marginal_benchmark/benchmark_metrics.csv`",
            "- `data_processed/marginal_benchmark/forecast_comparison_2026_2030.csv`",
            "- `data_processed/marginal_benchmark/selected_marginal_treatments.csv`",
            "- `reports/figures/marginal_benchmark_*.png` / `.pdf`",
            "",
            "## Final marginal envelopes",
            "",
            "Use these selections to generate final marginal uncertainty envelopes. Joint modeling starts only after the final P10/P50/P90 envelopes are reviewed and accepted.",
        ]
    )
    (REPORT_DIR / "marginal_model_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    data = load_inputs()
    series = build_series(data)
    metric_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []

    for spec, observed in series.values():
        x, y = clean_xy(spec, observed)
        series_metrics = benchmark_one_step(spec, x, y)
        series_forecasts = forecast_methods(spec, x, y)
        metric_rows.extend(series_metrics)
        forecast_rows.extend(series_forecasts)
        metric_df = pd.DataFrame(series_metrics)
        selected = choose_method(spec, metric_df)
        selected_rows.append(selected)
        meta_rows.append(
            {
                "series_id": spec.series_id,
                "observed_years": f"{int(x.min())}-{int(x.max())}",
                "n_observations": len(y),
                "unit": spec.unit,
                "benchmark_status": "scenario-constrained" if spec.series_id in LOW_EMISSION_SERIES else "benchmarked",
            }
        )
        plot_forecast_comparison(spec, x, y, pd.DataFrame(series_forecasts), str(selected["selected_method"]))

    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.DataFrame(forecast_rows)
    selected = pd.DataFrame(selected_rows)
    write_csv(
        OUT_DIR / "benchmark_metrics.csv",
        metric_rows,
        [
            "series_id",
            "method",
            "method_class",
            "n_observations",
            "n_backtests",
            "mae",
            "rmse",
            "mape_percent",
            "bias",
            "constraint_violations",
        ],
    )
    write_csv(
        OUT_DIR / "forecast_comparison_2026_2030.csv",
        forecast_rows,
        ["series_id", "method", "year", "forecast", "unit", "status"],
    )
    write_csv(
        OUT_DIR / "selected_marginal_treatments.csv",
        selected_rows,
        ["series_id", "selected_treatment", "selected_method", "reason"],
    )
    plot_mape_summary(metrics)
    write_report(meta_rows, metrics, forecasts, selected)
    print(f"Wrote benchmark outputs to {OUT_DIR}")
    print(f"Wrote benchmark report to {REPORT_DIR / 'marginal_model_benchmark.md'}")
    print(f"Wrote benchmark figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
