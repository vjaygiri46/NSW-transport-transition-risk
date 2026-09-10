"""Compute downside-risk, CVaR, and regret metrics."""

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
    DEPENDENCE_STRUCTURES,
    N_SAMPLES,
    RNG_SEED,
    SCENARIO_FAMILIES,
    add_risk_metrics,
    apply_style,
    load_inputs,
    save_fig,
    simulate_year,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "risk_decision"
REPORT_DIR = ROOT / "reports"

LOSS_WEIGHTS = {
    "weak_alignment": 0.15,
    "low_uptake_shortfall": 0.20,
    "renewable_shortfall": 0.15,
    "fossil_excess": 0.20,
    "emissions_excess": 0.20,
    "activity_excess": 0.10,
}


def normalized_excess(value: pd.Series, reference: float, low: float, high: float) -> pd.Series:
    denom = max(abs(high - low), 1e-9)
    return ((value - reference) / denom).clip(lower=0.0, upper=1.0)


def normalized_shortfall(value: pd.Series, reference: float, low: float, high: float) -> pd.Series:
    denom = max(abs(high - low), 1e-9)
    return ((reference - value) / denom).clip(lower=0.0, upper=1.0)


def add_loss(sim: pd.DataFrame, marginals: pd.DataFrame) -> pd.DataFrame:
    year = int(sim["year"].iloc[0])
    base = marginals[marginals["year"] == year].set_index("series_id")
    low_emission = sim["bev_fcev_share_nsw"] + sim["hybrid_share_nsw"]
    low_ref = float(base.loc["bev_fcev_share_nsw", "p50"] + base.loc["hybrid_share_nsw", "p50"])
    low_min = float(base.loc["bev_fcev_share_nsw", "p10"] + base.loc["hybrid_share_nsw", "p10"])
    low_max = float(base.loc["bev_fcev_share_nsw", "p90"] + base.loc["hybrid_share_nsw", "p90"])

    renewable_ref = float(base.loc["renewable_share_nsw", "p50"])
    renewable_min = float(base.loc["renewable_share_nsw", "p10"])
    renewable_max = float(base.loc["renewable_share_nsw", "p90"])

    fossil = sim["petrol_share_nsw"] + sim["diesel_share_nsw"]
    fossil_ref = float(base.loc["petrol_share_nsw", "p50"] + base.loc["diesel_share_nsw", "p50"])
    fossil_min = float(base.loc["petrol_share_nsw", "p10"] + base.loc["diesel_share_nsw", "p10"])
    fossil_max = float(base.loc["petrol_share_nsw", "p90"] + base.loc["diesel_share_nsw", "p90"])

    emissions_ref = float(base.loc["road_direct_emissions_national", "p50"])
    emissions_min = float(base.loc["road_direct_emissions_national", "p10"])
    emissions_max = float(base.loc["road_direct_emissions_national", "p90"])

    activity_ref = float(base.loc["sydney_passenger_activity", "p50"])
    activity_min = float(base.loc["sydney_passenger_activity", "p10"])
    activity_max = float(base.loc["sydney_passenger_activity", "p90"])

    out = sim.copy()
    out["low_uptake_shortfall"] = normalized_shortfall(low_emission, low_ref, low_min, low_max)
    out["renewable_shortfall"] = normalized_shortfall(out["renewable_share_nsw"], renewable_ref, renewable_min, renewable_max)
    out["fossil_excess"] = normalized_excess(fossil, fossil_ref, fossil_min, fossil_max)
    out["emissions_excess"] = normalized_excess(
        out["road_direct_emissions_national"], emissions_ref, emissions_min, emissions_max
    )
    out["activity_excess"] = normalized_excess(
        out["sydney_passenger_activity"], activity_ref, activity_min, activity_max
    )
    out["transition_loss"] = (
        LOSS_WEIGHTS["weak_alignment"] * out["weak_alignment"].astype(float)
        + LOSS_WEIGHTS["low_uptake_shortfall"] * out["low_uptake_shortfall"]
        + LOSS_WEIGHTS["renewable_shortfall"] * out["renewable_shortfall"]
        + LOSS_WEIGHTS["fossil_excess"] * out["fossil_excess"]
        + LOSS_WEIGHTS["emissions_excess"] * out["emissions_excess"]
        + LOSS_WEIGHTS["activity_excess"] * out["activity_excess"]
    )
    return out


def summarize_loss(sim: pd.DataFrame) -> dict[str, object]:
    loss = sim["transition_loss"].to_numpy(dtype=float)
    threshold = float(np.quantile(loss, 0.90))
    tail = loss[loss >= threshold]
    return {
        "year": int(sim["year"].iloc[0]),
        "dependence": sim["dependence"].iloc[0],
        "scenario_family": sim["scenario_family"].iloc[0],
        "n_samples": len(sim),
        "mean_loss": float(np.mean(loss)),
        "p50_loss": float(np.quantile(loss, 0.50)),
        "p90_loss": threshold,
        "cvar90_loss": float(np.mean(tail)) if len(tail) else threshold,
        "weak_alignment_probability": float(sim["weak_alignment"].mean()),
        "fossil_persistence_probability": float(sim["fossil_persistence"].mean()),
        "emissions_pressure_probability": float(sim["emissions_pressure"].mean()),
        "activity_pressure_probability": float(sim["activity_pressure"].mean()),
    }


def compute_regret(loss_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (year, dependence), group in loss_summary.groupby(["year", "dependence"]):
        best = float(group["mean_loss"].min())
        for _, row in group.iterrows():
            rows.append(
                {
                    "year": int(year),
                    "dependence": dependence,
                    "scenario_family": row["scenario_family"],
                    "mean_loss": float(row["mean_loss"]),
                    "regret_vs_best_scenario": float(row["mean_loss"] - best),
                }
            )
    return pd.DataFrame(rows)


def plot_cvar_bar(loss_summary: pd.DataFrame) -> None:
    final = loss_summary[loss_summary["year"] == 2030].sort_values(["scenario_family", "dependence"])
    labels = [f"{d}\n{s}" for d, s in zip(final["dependence"], final["scenario_family"])]
    fig, ax = plt.subplots(figsize=(7.0, 2.625))
    ax.bar(np.arange(len(final)), final["cvar90_loss"], color="tab:red", alpha=0.75)
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("CVaR90-style loss")
    ax.set_title("2030 downside tail risk")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "risk_cvar90_2030")


def plot_loss_heatmap(loss_summary: pd.DataFrame) -> None:
    final = loss_summary[loss_summary["year"] == 2030]
    pivot = final.pivot(index="scenario_family", columns="dependence", values="mean_loss")
    pivot = pivot.loc[SCENARIO_FAMILIES, DEPENDENCE_STRUCTURES]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="magma", vmin=0, vmax=max(0.01, float(pivot.max().max())))
    ax.set_xticks(np.arange(len(DEPENDENCE_STRUCTURES)))
    ax.set_xticklabels(["Independent", "Gaussian", "Student-t"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIO_FAMILIES)))
    ax.set_yticklabels(["Conservative", "Central", "Optimistic", "Stress tail"])
    for i in range(len(SCENARIO_FAMILIES)):
        for j in range(len(DEPENDENCE_STRUCTURES)):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white")
    ax.set_title("2030 mean transition loss")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "risk_mean_loss_heatmap_2030")


def plot_component_bars(component_summary: pd.DataFrame) -> None:
    final = component_summary[
        (component_summary["year"] == 2030)
        & (component_summary["dependence"] == "student_t")
        & (component_summary["scenario_family"].isin(["central", "stress_tail"]))
    ].copy()
    components = [
        "low_uptake_shortfall",
        "renewable_shortfall",
        "fossil_excess",
        "emissions_excess",
        "activity_excess",
    ]
    labels = ["Low uptake", "Renewable shortfall", "Fossil excess", "Emissions excess", "Activity excess"]
    x = np.arange(len(components))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 2.625))
    for offset, scenario in [(-width / 2, "central"), (width / 2, "stress_tail")]:
        row = final[final["scenario_family"] == scenario].iloc[0]
        values = [float(row[col]) for col in components]
        ax.bar(x + offset, values, width=width, label=scenario.replace("_", " ").title(), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("Mean normalized component")
    ax.set_title("Student-t loss components, 2030")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "risk_loss_components_student_t_2030")


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


def write_report(loss_summary: pd.DataFrame, regret: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = loss_summary[loss_summary["year"] == 2030].sort_values(["scenario_family", "dependence"])
    cols = ["dependence", "scenario_family", "mean_loss", "p90_loss", "cvar90_loss"]
    lines = [
        "# Risk Decision Metrics",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This step converts the joint Monte Carlo outputs into downside-risk metrics. It evaluates the scenarios rather than optimizing the forecasts.",
        "",
        "## Transition Loss Definition",
        "",
        "The transition-loss index combines weak alignment, low uptake, renewable shortfall, fossil excess, emissions excess, and activity excess. Components are normalized against each year's final marginal envelope.",
        "",
        "| Component | Weight |",
        "|---|---:|",
    ]
    for key, value in LOSS_WEIGHTS.items():
        lines.append(f"| `{key}` | {value:.2f} |")
    lines.extend(["", "## 2030 Downside-Risk Summary", ""])
    lines.extend(markdown_table(final[cols], cols))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Weak alignment alone misses conservative and stress-tail risk because those scenarios have low uptake rather than high-uptake/low-renewable mismatch.",
            "- The transition-loss index captures low uptake, renewable shortfall, fossil persistence, emissions pressure, and activity pressure together.",
            "- CVaR90-style loss is the mean loss in the worst 10 percent of simulated samples for each scenario/dependence case.",
            "- Regret here compares scenario-family mean loss against the best scenario under the same year and dependence structure; it does not optimize intervention policy yet.",
            "",
            "## Outputs",
            "",
            "- `data_processed/risk_decision/risk_metric_summary.csv`",
            "- `data_processed/risk_decision/regret_summary.csv`",
            "- `data_processed/risk_decision/loss_component_summary.csv`",
            "- `reports/figures/risk_cvar90_2030.png` / `.pdf`",
            "- `reports/figures/risk_mean_loss_heatmap_2030.png` / `.pdf`",
            "- `reports/figures/risk_loss_components_student_t_2030.png` / `.pdf`",
        ]
    )
    (REPORT_DIR / "risk_decision_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    marginals, corr = load_inputs()
    loss_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
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
                sim = add_loss(sim, marginals)
                loss_rows.append(summarize_loss(sim))
                component_rows.append(
                    {
                        "year": int(year),
                        "dependence": dependence,
                        "scenario_family": scenario,
                        "low_uptake_shortfall": float(sim["low_uptake_shortfall"].mean()),
                        "renewable_shortfall": float(sim["renewable_shortfall"].mean()),
                        "fossil_excess": float(sim["fossil_excess"].mean()),
                        "emissions_excess": float(sim["emissions_excess"].mean()),
                        "activity_excess": float(sim["activity_excess"].mean()),
                    }
                )

    loss_summary = pd.DataFrame(loss_rows)
    component_summary = pd.DataFrame(component_rows)
    regret = compute_regret(loss_summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_summary.to_csv(OUT_DIR / "risk_metric_summary.csv", index=False)
    regret.to_csv(OUT_DIR / "regret_summary.csv", index=False)
    component_summary.to_csv(OUT_DIR / "loss_component_summary.csv", index=False)
    plot_cvar_bar(loss_summary)
    plot_loss_heatmap(loss_summary)
    plot_component_bars(component_summary)
    write_report(loss_summary, regret)
    print(f"Wrote risk-decision outputs to {OUT_DIR}")
    print(f"Wrote risk-decision report to {REPORT_DIR / 'risk_decision_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
