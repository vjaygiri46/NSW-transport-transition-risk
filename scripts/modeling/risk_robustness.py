"""Run sensitivity checks for the transition-risk metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from joint_monte_carlo import (  # noqa: E402
    DEPENDENCE_LABELS,
    DEPENDENCE_STRUCTURES,
    RNG_SEED,
    SCENARIO_FAMILIES,
    SCENARIO_LABELS,
    add_risk_metrics,
    apply_style,
    load_inputs,
    save_fig,
    simulate_year,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "risk_robustness"
REPORT_DIR = ROOT / "reports"

COMPONENTS = [
    "weak_alignment",
    "low_uptake_shortfall",
    "renewable_shortfall",
    "fossil_excess",
    "emissions_excess",
    "activity_excess",
]

WEIGHT_SCHEMES = {
    "baseline": {
        "weak_alignment": 0.15,
        "low_uptake_shortfall": 0.20,
        "renewable_shortfall": 0.15,
        "fossil_excess": 0.20,
        "emissions_excess": 0.20,
        "activity_excess": 0.10,
    },
    "equal": {
        "weak_alignment": 1 / 6,
        "low_uptake_shortfall": 1 / 6,
        "renewable_shortfall": 1 / 6,
        "fossil_excess": 1 / 6,
        "emissions_excess": 1 / 6,
        "activity_excess": 1 / 6,
    },
    "uptake_heavy": {
        "weak_alignment": 0.10,
        "low_uptake_shortfall": 0.35,
        "renewable_shortfall": 0.15,
        "fossil_excess": 0.15,
        "emissions_excess": 0.15,
        "activity_excess": 0.10,
    },
    "grid_emissions_heavy": {
        "weak_alignment": 0.10,
        "low_uptake_shortfall": 0.15,
        "renewable_shortfall": 0.25,
        "fossil_excess": 0.10,
        "emissions_excess": 0.30,
        "activity_excess": 0.10,
    },
    "fossil_heavy": {
        "weak_alignment": 0.10,
        "low_uptake_shortfall": 0.15,
        "renewable_shortfall": 0.10,
        "fossil_excess": 0.35,
        "emissions_excess": 0.20,
        "activity_excess": 0.10,
    },
}

THRESHOLD_SHIFTS = {
    "lenient": -0.25,
    "baseline": 0.0,
    "strict": 0.25,
}

CVAR_LEVELS = [0.90, 0.95]


def normalized_excess(value: pd.Series, reference: float, low: float, high: float) -> pd.Series:
    denom = max(abs(high - low), 1e-9)
    return ((value - reference) / denom).clip(lower=0.0, upper=1.0)


def normalized_shortfall(value: pd.Series, reference: float, low: float, high: float) -> pd.Series:
    denom = max(abs(high - low), 1e-9)
    return ((reference - value) / denom).clip(lower=0.0, upper=1.0)


def shifted_reference(p10: float, p50: float, p90: float, shift: float, *, beneficial: bool) -> float:
    if shift == 0:
        return p50
    if beneficial:
        return p50 + shift * (p90 - p50 if shift > 0 else p50 - p10)
    return p50 - shift * (p50 - p10 if shift > 0 else p90 - p50)


def add_loss_components(sim: pd.DataFrame, marginals: pd.DataFrame, threshold_mode: str) -> pd.DataFrame:
    year = int(sim["year"].iloc[0])
    shift = THRESHOLD_SHIFTS[threshold_mode]
    base = marginals[marginals["year"] == year].set_index("series_id")

    low_emission = sim["bev_fcev_share_nsw"] + sim["hybrid_share_nsw"]
    low_p10 = float(base.loc["bev_fcev_share_nsw", "p10"] + base.loc["hybrid_share_nsw", "p10"])
    low_p50 = float(base.loc["bev_fcev_share_nsw", "p50"] + base.loc["hybrid_share_nsw", "p50"])
    low_p90 = float(base.loc["bev_fcev_share_nsw", "p90"] + base.loc["hybrid_share_nsw", "p90"])
    low_ref = shifted_reference(low_p10, low_p50, low_p90, shift, beneficial=True)

    renewable_p10 = float(base.loc["renewable_share_nsw", "p10"])
    renewable_p50 = float(base.loc["renewable_share_nsw", "p50"])
    renewable_p90 = float(base.loc["renewable_share_nsw", "p90"])
    renewable_ref = shifted_reference(renewable_p10, renewable_p50, renewable_p90, shift, beneficial=True)

    fossil = sim["petrol_share_nsw"] + sim["diesel_share_nsw"]
    fossil_p10 = float(base.loc["petrol_share_nsw", "p10"] + base.loc["diesel_share_nsw", "p10"])
    fossil_p50 = float(base.loc["petrol_share_nsw", "p50"] + base.loc["diesel_share_nsw", "p50"])
    fossil_p90 = float(base.loc["petrol_share_nsw", "p90"] + base.loc["diesel_share_nsw", "p90"])
    fossil_ref = shifted_reference(fossil_p10, fossil_p50, fossil_p90, shift, beneficial=False)

    emissions_p10 = float(base.loc["road_direct_emissions_national", "p10"])
    emissions_p50 = float(base.loc["road_direct_emissions_national", "p50"])
    emissions_p90 = float(base.loc["road_direct_emissions_national", "p90"])
    emissions_ref = shifted_reference(emissions_p10, emissions_p50, emissions_p90, shift, beneficial=False)

    activity_p10 = float(base.loc["sydney_passenger_activity", "p10"])
    activity_p50 = float(base.loc["sydney_passenger_activity", "p50"])
    activity_p90 = float(base.loc["sydney_passenger_activity", "p90"])
    activity_ref = shifted_reference(activity_p10, activity_p50, activity_p90, shift, beneficial=False)

    bev_ref = shifted_reference(
        float(base.loc["bev_fcev_share_nsw", "p10"]),
        float(base.loc["bev_fcev_share_nsw", "p50"]),
        float(base.loc["bev_fcev_share_nsw", "p90"]),
        shift,
        beneficial=True,
    )

    out = sim.copy()
    out["weak_alignment"] = (out["bev_fcev_share_nsw"] >= bev_ref) & (out["renewable_share_nsw"] < renewable_ref)
    out["low_uptake_shortfall"] = normalized_shortfall(low_emission, low_ref, low_p10, low_p90)
    out["renewable_shortfall"] = normalized_shortfall(out["renewable_share_nsw"], renewable_ref, renewable_p10, renewable_p90)
    out["fossil_excess"] = normalized_excess(fossil, fossil_ref, fossil_p10, fossil_p90)
    out["emissions_excess"] = normalized_excess(
        out["road_direct_emissions_national"], emissions_ref, emissions_p10, emissions_p90
    )
    out["activity_excess"] = normalized_excess(out["sydney_passenger_activity"], activity_ref, activity_p10, activity_p90)
    return out


def transition_loss(sim: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    loss = np.zeros(len(sim), dtype=float)
    for component, weight in weights.items():
        loss += weight * sim[component].astype(float).to_numpy()
    return loss


def summarize_variant(
    sim: pd.DataFrame,
    *,
    weight_scheme: str,
    threshold_mode: str,
    cvar_level: float,
) -> dict[str, object]:
    loss = transition_loss(sim, WEIGHT_SCHEMES[weight_scheme])
    threshold = float(np.quantile(loss, cvar_level))
    tail = loss[loss >= threshold]
    return {
        "year": int(sim["year"].iloc[0]),
        "dependence": sim["dependence"].iloc[0],
        "scenario_family": sim["scenario_family"].iloc[0],
        "weight_scheme": weight_scheme,
        "threshold_mode": threshold_mode,
        "cvar_level": cvar_level,
        "mean_loss": float(np.mean(loss)),
        "p_tail_loss": threshold,
        "cvar_loss": float(np.mean(tail)) if len(tail) else threshold,
    }


def scenario_stability(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    final = summary[summary["year"] == 2030]
    for keys, group in final.groupby(["dependence", "weight_scheme", "threshold_mode", "cvar_level"]):
        top_mean = group.sort_values("mean_loss", ascending=False).iloc[0]
        top_cvar = group.sort_values("cvar_loss", ascending=False).iloc[0]
        rows.append(
            {
                "dependence": keys[0],
                "weight_scheme": keys[1],
                "threshold_mode": keys[2],
                "cvar_level": keys[3],
                "top_mean_loss_scenario": top_mean["scenario_family"],
                "top_cvar_loss_scenario": top_cvar["scenario_family"],
            }
        )
    return pd.DataFrame(rows)


def plot_top_risk_frequency(stability: pd.DataFrame) -> None:
    counts = stability["top_cvar_loss_scenario"].value_counts().reindex(SCENARIO_FAMILIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.bar(np.arange(len(counts)), counts.to_numpy(), color="tab:red", alpha=0.78)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels([SCENARIO_LABELS[x] for x in counts.index], rotation=25, ha="right")
    ax.set_ylabel("Count across variants")
    ax.set_title("Top CVaR-risk scenario, 2030")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "robustness_top_cvar_scenario_frequency_2030")


def plot_baseline_cvar_sensitivity(summary: pd.DataFrame) -> None:
    final = summary[
        (summary["year"] == 2030)
        & (summary["weight_scheme"] == "baseline")
        & (summary["cvar_level"] == 0.95)
    ].copy()
    pivot = final.pivot_table(
        index="scenario_family",
        columns="threshold_mode",
        values="cvar_loss",
        aggfunc="mean",
    ).loc[SCENARIO_FAMILIES, ["lenient", "baseline", "strict"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="magma", vmin=0, vmax=max(0.01, float(pivot.max().max())))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Lenient", "Baseline", "Strict"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIO_FAMILIES)))
    ax.set_yticklabels([SCENARIO_LABELS[x] for x in SCENARIO_FAMILIES])
    for i in range(len(SCENARIO_FAMILIES)):
        for j in range(3):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white")
    ax.set_title("CVaR95 sensitivity, 2030")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "robustness_cvar95_threshold_sensitivity_2030")


def plot_central_dependence_range(summary: pd.DataFrame) -> None:
    final = summary[
        (summary["year"] == 2030)
        & (summary["scenario_family"] == "central")
        & (summary["threshold_mode"] == "baseline")
        & (summary["cvar_level"] == 0.95)
    ].copy()
    pivot = final.pivot_table(index="weight_scheme", columns="dependence", values="cvar_loss", aggfunc="mean")
    pivot = pivot.loc[list(WEIGHT_SCHEMES), DEPENDENCE_STRUCTURES]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=max(0.01, float(pivot.max().max())))
    ax.set_xticks(np.arange(len(DEPENDENCE_STRUCTURES)))
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in DEPENDENCE_STRUCTURES], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(WEIGHT_SCHEMES)))
    ax.set_yticklabels([x.replace("_", " ").title() for x in WEIGHT_SCHEMES])
    for i in range(len(WEIGHT_SCHEMES)):
        for j in range(len(DEPENDENCE_STRUCTURES)):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white")
    ax.set_title("Central CVaR95 by weights")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "robustness_central_dependence_cvar95_2030")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    final = summary[summary["year"] == 2030]
    mean_frequency = (
        stability["top_mean_loss_scenario"]
        .value_counts()
        .reindex(SCENARIO_FAMILIES, fill_value=0)
        .rename_axis("scenario_family")
        .reset_index(name="top_mean_count")
    )
    cvar_frequency = (
        stability["top_cvar_loss_scenario"]
        .value_counts()
        .reindex(SCENARIO_FAMILIES, fill_value=0)
        .rename_axis("scenario_family")
        .reset_index(name="top_cvar_count")
    )
    baseline = final[
        (final["weight_scheme"] == "baseline")
        & (final["threshold_mode"] == "baseline")
        & (final["cvar_level"] == 0.95)
    ].sort_values(["scenario_family", "dependence"])
    lines = [
        "# Risk Metric Robustness",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This analysis tests whether the risk conclusions are stable under changes to weights, threshold strictness, and tail-risk level.",
        "",
        "## Sensitivity Grid",
        "",
        f"- Weight schemes: {', '.join(WEIGHT_SCHEMES)}.",
        "- Threshold modes: lenient, baseline, strict.",
        "- Tail-risk levels: CVaR90-style and CVaR95-style.",
        "",
        "## 2030 Top Mean-Risk Frequency",
        "",
    ]
    lines.extend(markdown_table(mean_frequency, ["scenario_family", "top_mean_count"]))
    lines.extend(
        [
            "",
        "## 2030 Top CVaR-Risk Frequency",
        "",
        ]
    )
    lines.extend(markdown_table(cvar_frequency, ["scenario_family", "top_cvar_count"]))
    lines.extend(
        [
            "",
            "## Baseline CVaR95-Style Loss, 2030",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            baseline[["dependence", "scenario_family", "mean_loss", "p_tail_loss", "cvar_loss"]],
            ["dependence", "scenario_family", "mean_loss", "p_tail_loss", "cvar_loss"],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stress/tail is the most stable high-risk case by mean loss: it is the highest-mean-loss scenario in all tested variants.",
            "- Stress/tail is also the dominant tail-risk case: it is the highest-CVaR scenario in most variants, while conservative becomes highest only in a small number of lenient-threshold variants.",
            "- Conservative remains clearly adverse by mean loss, but central/reference can show comparable or higher tail loss in some CVaR95-style cases because central samples the full uncertainty envelope.",
            "- Central/reference retains non-zero downside tail loss, especially under CVaR95-style evaluation.",
            "- Optimistic remains near zero because it is constructed as the favorable boundary case under the current loss definition.",
            "- Dependence choice changes magnitudes, but it does not overturn the scenario-family ranking in this sensitivity pass.",
            "",
            "## Outputs",
            "",
            "- `data_processed/risk_robustness/robustness_summary.csv`",
            "- `data_processed/risk_robustness/scenario_stability.csv`",
            "- `reports/figures/robustness_top_cvar_scenario_frequency_2030.png` / `.pdf`",
            "- `reports/figures/robustness_cvar95_threshold_sensitivity_2030.png` / `.pdf`",
            "- `reports/figures/robustness_central_dependence_cvar95_2030.png` / `.pdf`",
        ]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "risk_metric_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    marginals, corr = load_inputs()
    rows: list[dict[str, object]] = []
    for dependence in DEPENDENCE_STRUCTURES:
        for scenario in SCENARIO_FAMILIES:
            for year in sorted(marginals["year"].unique()):
                sim = simulate_year(
                    marginals,
                    corr,
                    year=int(year),
                    dependence=dependence,
                    scenario=scenario,
                    rng=rng,
                )
                sim = add_risk_metrics(sim, marginals)
                for threshold_mode in THRESHOLD_SHIFTS:
                    adjusted = add_loss_components(sim, marginals, threshold_mode)
                    for weight_scheme in WEIGHT_SCHEMES:
                        for cvar_level in CVAR_LEVELS:
                            rows.append(
                                summarize_variant(
                                    adjusted,
                                    weight_scheme=weight_scheme,
                                    threshold_mode=threshold_mode,
                                    cvar_level=cvar_level,
                                )
                            )
    summary = pd.DataFrame(rows)
    stability = scenario_stability(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    stability.to_csv(OUT_DIR / "scenario_stability.csv", index=False)
    plot_top_risk_frequency(stability)
    plot_baseline_cvar_sensitivity(summary)
    plot_central_dependence_range(summary)
    write_report(summary, stability)
    print(f"Wrote robustness outputs to {OUT_DIR}")
    print(f"Wrote robustness report to {REPORT_DIR / 'risk_metric_robustness.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
