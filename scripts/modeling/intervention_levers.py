"""Evaluate bounded intervention-scenario levers against transition-risk metrics."""

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
    add_risk_metrics,
    apply_style,
    load_inputs,
    save_fig,
    simulate_year,
)
from risk_decision_metrics import add_loss  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "intervention_levers"
REPORT_DIR = ROOT / "reports"

SCENARIOS = ["central", "conservative", "stress_tail"]
LEVER_LABELS = {
    "none": "No lever",
    "uptake_push": "Uptake push",
    "renewable_push": "Renewable push",
    "fossil_reduction": "Fossil reduction",
    "balanced_package": "Balanced package",
    "delay_case": "Delay case",
}

LEVER_SPECS = {
    "none": {},
    "uptake_push": {
        "bev_fcev_share_nsw": ("toward_p90", 0.35),
        "hybrid_share_nsw": ("toward_p90", 0.20),
        "petrol_share_nsw": ("toward_p10", 0.18),
        "diesel_share_nsw": ("toward_p10", 0.10),
    },
    "renewable_push": {
        "renewable_share_nsw": ("toward_p90", 0.35),
        "road_direct_emissions_national": ("toward_p10", 0.12),
    },
    "fossil_reduction": {
        "petrol_share_nsw": ("toward_p10", 0.30),
        "diesel_share_nsw": ("toward_p10", 0.25),
        "road_energy_use_national": ("toward_p10", 0.10),
        "road_direct_emissions_national": ("toward_p10", 0.15),
    },
    "balanced_package": {
        "bev_fcev_share_nsw": ("toward_p90", 0.25),
        "hybrid_share_nsw": ("toward_p90", 0.15),
        "renewable_share_nsw": ("toward_p90", 0.25),
        "petrol_share_nsw": ("toward_p10", 0.20),
        "diesel_share_nsw": ("toward_p10", 0.15),
        "road_energy_use_national": ("toward_p10", 0.08),
        "road_direct_emissions_national": ("toward_p10", 0.15),
    },
    "delay_case": {
        "bev_fcev_share_nsw": ("toward_p10", 0.20),
        "hybrid_share_nsw": ("toward_p10", 0.15),
        "renewable_share_nsw": ("toward_p10", 0.20),
        "petrol_share_nsw": ("toward_p90", 0.15),
        "diesel_share_nsw": ("toward_p90", 0.10),
        "road_direct_emissions_national": ("toward_p90", 0.12),
    },
}


def move_within_envelope(values: pd.Series, p10: float, p90: float, direction: str, strength: float) -> pd.Series:
    if direction == "toward_p90":
        return values + strength * (p90 - values)
    if direction == "toward_p10":
        return values - strength * (values - p10)
    raise ValueError(f"Unknown lever direction: {direction}")


def apply_lever(sim: pd.DataFrame, marginals: pd.DataFrame, lever: str) -> pd.DataFrame:
    out = sim.copy()
    if lever == "none":
        out["lever"] = lever
        return out

    year = int(sim["year"].iloc[0])
    base = marginals[marginals["year"] == year].set_index("series_id")
    for series_id, (direction, strength) in LEVER_SPECS[lever].items():
        row = base.loc[series_id]
        out[series_id] = move_within_envelope(
            out[series_id],
            float(row["p10"]),
            float(row["p90"]),
            direction,
            float(strength),
        ).clip(lower=float(row["p10"]), upper=float(row["p90"]))
    out["lever"] = lever
    return out


def summarize(sim: pd.DataFrame) -> dict[str, object]:
    loss = sim["transition_loss"].to_numpy(dtype=float)
    p90 = float(np.quantile(loss, 0.90))
    p95 = float(np.quantile(loss, 0.95))
    return {
        "year": int(sim["year"].iloc[0]),
        "dependence": sim["dependence"].iloc[0],
        "scenario_family": sim["scenario_family"].iloc[0],
        "lever": sim["lever"].iloc[0],
        "n_samples": len(sim),
        "mean_loss": float(np.mean(loss)),
        "p90_loss": p90,
        "cvar90_loss": float(np.mean(loss[loss >= p90])),
        "p95_loss": p95,
        "cvar95_loss": float(np.mean(loss[loss >= p95])),
        "weak_alignment_probability": float(sim["weak_alignment"].mean()),
        "fossil_persistence_probability": float(sim["fossil_persistence"].mean()),
        "emissions_pressure_probability": float(sim["emissions_pressure"].mean()),
        "activity_pressure_probability": float(sim["activity_pressure"].mean()),
    }


def summarize_components(sim: pd.DataFrame) -> dict[str, object]:
    return {
        "year": int(sim["year"].iloc[0]),
        "dependence": sim["dependence"].iloc[0],
        "scenario_family": sim["scenario_family"].iloc[0],
        "lever": sim["lever"].iloc[0],
        "weak_alignment_probability": float(sim["weak_alignment"].mean()),
        "low_uptake_shortfall": float(sim["low_uptake_shortfall"].mean()),
        "renewable_shortfall": float(sim["renewable_shortfall"].mean()),
        "fossil_excess": float(sim["fossil_excess"].mean()),
        "emissions_excess": float(sim["emissions_excess"].mean()),
        "activity_excess": float(sim["activity_excess"].mean()),
    }


def add_reductions(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key_cols = ["year", "dependence", "scenario_family"]
    for keys, group in summary.groupby(key_cols):
        base = group[group["lever"] == "none"].iloc[0]
        for _, row in group.iterrows():
            out = row.to_dict()
            out["mean_loss_reduction_vs_none"] = float(base["mean_loss"] - row["mean_loss"])
            out["cvar95_reduction_vs_none"] = float(base["cvar95_loss"] - row["cvar95_loss"])
            rows.append(out)
    return pd.DataFrame(rows)


def robust_ranking(reductions: pd.DataFrame) -> pd.DataFrame:
    final = reductions[(reductions["year"] == 2030) & (reductions["lever"] != "none")].copy()
    rows: list[dict[str, object]] = []
    for lever, group in final.groupby("lever"):
        rows.append(
            {
                "lever": lever,
                "mean_reduction_median": float(group["mean_loss_reduction_vs_none"].median()),
                "mean_reduction_min": float(group["mean_loss_reduction_vs_none"].min()),
                "cvar95_reduction_median": float(group["cvar95_reduction_vs_none"].median()),
                "cvar95_reduction_min": float(group["cvar95_reduction_vs_none"].min()),
                "share_positive_mean_reduction": float((group["mean_loss_reduction_vs_none"] > 0).mean()),
                "share_positive_cvar95_reduction": float((group["cvar95_reduction_vs_none"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cvar95_reduction_median", "mean_reduction_median"], ascending=False)


def plot_mean_reduction(reductions: pd.DataFrame) -> None:
    final = reductions[
        (reductions["year"] == 2030)
        & (reductions["scenario_family"] == "stress_tail")
        & (reductions["lever"] != "none")
    ].copy()
    pivot = final.pivot_table(index="lever", columns="dependence", values="mean_loss_reduction_vs_none", aggfunc="mean")
    order = ["balanced_package", "fossil_reduction", "uptake_push", "renewable_push", "delay_case"]
    pivot = pivot.loc[order, DEPENDENCE_STRUCTURES]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", vmin=-0.08, vmax=0.16)
    ax.set_xticks(np.arange(len(DEPENDENCE_STRUCTURES)))
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in DEPENDENCE_STRUCTURES], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LEVER_LABELS[x] for x in order])
    for i in range(len(order)):
        for j in range(len(DEPENDENCE_STRUCTURES)):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title("Stress-tail mean-loss reduction, 2030")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "intervention_stress_tail_mean_reduction_2030")


def plot_cvar_reduction(reductions: pd.DataFrame) -> None:
    final = reductions[
        (reductions["year"] == 2030)
        & (reductions["scenario_family"].isin(["central", "conservative", "stress_tail"]))
        & (reductions["lever"] != "none")
    ].copy()
    pivot = final.pivot_table(index="lever", columns="scenario_family", values="cvar95_reduction_vs_none", aggfunc="median")
    order = ["balanced_package", "fossil_reduction", "uptake_push", "renewable_push", "delay_case"]
    pivot = pivot.loc[order, ["central", "conservative", "stress_tail"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", vmin=-0.10, vmax=0.18)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Central", "Conservative", "Stress tail"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LEVER_LABELS[x] for x in order])
    for i in range(len(order)):
        for j in range(3):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title("Median CVaR95 reduction, 2030")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "intervention_cvar95_reduction_2030")


def plot_robust_ranking(ranking: pd.DataFrame) -> None:
    plot_df = ranking[ranking["lever"] != "delay_case"].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["cvar95_reduction_median"], color="tab:blue", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LEVER_LABELS[x] for x in plot_df["lever"]])
    ax.set_xlabel("Median CVaR95 reduction")
    ax.set_title("Robust lever ranking, 2030")
    ax.tick_params(labelsize=6)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "intervention_robust_ranking_2030")


def plot_component_reduction(component_summary: pd.DataFrame) -> None:
    final = component_summary[
        (component_summary["year"] == 2030)
        & (component_summary["scenario_family"] == "stress_tail")
        & (component_summary["dependence"] == "student_t")
    ].copy()
    components = [
        "weak_alignment_probability",
        "low_uptake_shortfall",
        "renewable_shortfall",
        "fossil_excess",
        "emissions_excess",
    ]
    labels = ["Weak align.", "Low uptake", "Renewable", "Fossil", "Emissions"]
    base = final[final["lever"] == "none"].iloc[0]
    order = ["balanced_package", "fossil_reduction", "renewable_push", "uptake_push", "delay_case"]
    matrix = []
    for lever in order:
        row = final[final["lever"] == lever].iloc[0]
        matrix.append([float(base[col] - row[col]) for col in components])
    values = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.0, 2.625))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-0.35, vmax=0.35)
    ax.set_xticks(np.arange(len(components)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LEVER_LABELS[x] for x in order])
    for i in range(len(order)):
        for j in range(len(components)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title("Component reduction, Student-t stress-tail 2030")
    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "intervention_component_reduction_student_t_stress_tail_2030")


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


def write_report(summary: pd.DataFrame, ranking: pd.DataFrame, component_summary: pd.DataFrame) -> None:
    final_ranking = ranking.copy()
    final_ranking["lever"] = final_ranking["lever"].map(LEVER_LABELS)
    stress = summary[
        (summary["year"] == 2030)
        & (summary["scenario_family"] == "stress_tail")
        & (summary["dependence"] == "student_t")
    ].sort_values("cvar95_loss")
    stress = stress.copy()
    stress["lever"] = stress["lever"].map(LEVER_LABELS)
    components = component_summary[
        (component_summary["year"] == 2030)
        & (component_summary["scenario_family"] == "stress_tail")
        & (component_summary["dependence"] == "student_t")
    ].sort_values("lever")
    components = components.copy()
    components["lever"] = components["lever"].map(LEVER_LABELS)
    lines = [
        "# Intervention Lever Scenarios",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This stage evaluates bounded counterfactual driver shifts against the transition-loss metrics. The levers are not causal policy-effect estimates.",
        "",
        "## Lever Definition",
        "",
        "- Each lever moves selected simulated variables toward the favorable side of their existing marginal envelope.",
        "- The delay case moves selected variables toward the adverse side and is included as a comparison case.",
        "- No variable is moved outside its modeled P10-P90 envelope.",
        "",
        "## Robust 2030 Ranking",
        "",
    ]
    rank_cols = [
        "lever",
        "mean_reduction_median",
        "mean_reduction_min",
        "cvar95_reduction_median",
        "cvar95_reduction_min",
        "share_positive_cvar95_reduction",
    ]
    lines.extend(markdown_table(final_ranking[rank_cols], rank_cols))
    lines.extend(
        [
            "",
            "## Student-t Stress-Tail Case, 2030",
            "",
        ]
    )
    stress_cols = ["lever", "mean_loss", "cvar90_loss", "cvar95_loss"]
    lines.extend(markdown_table(stress[stress_cols], stress_cols))
    lines.extend(
        [
            "",
            "## Student-t Stress-Tail Components, 2030",
            "",
        ]
    )
    component_cols = [
        "lever",
        "weak_alignment_probability",
        "low_uptake_shortfall",
        "renewable_shortfall",
        "fossil_excess",
        "emissions_excess",
    ]
    lines.extend(markdown_table(components[component_cols], component_cols))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Balanced package gives the largest stress-tail reduction because it improves uptake, renewable alignment, fossil persistence, and emissions pressure together.",
            "- Fossil reduction and renewable push are more robust than uptake-only under the current loss definition because they reduce tail loss in every tested 2030 case.",
            "- Uptake-only is not robust: it can reduce mean loss but worsen tail loss when uptake improves without matching renewable and fossil/emissions improvement.",
            "- Delay case increases risk and is retained as an adverse comparison rather than a recommended action.",
            "- These are bounded driver-shift tests; they should be interpreted as sensitivity of risk to improved transition drivers, not as proof of policy causality.",
            "",
            "## Outputs",
            "",
            "- `data_processed/intervention_levers/intervention_summary.csv`",
            "- `data_processed/intervention_levers/intervention_reductions.csv`",
            "- `data_processed/intervention_levers/intervention_robust_ranking.csv`",
            "- `data_processed/intervention_levers/intervention_component_summary.csv`",
            "- `reports/figures/intervention_stress_tail_mean_reduction_2030.png` / `.pdf`",
            "- `reports/figures/intervention_cvar95_reduction_2030.png` / `.pdf`",
            "- `reports/figures/intervention_robust_ranking_2030.png` / `.pdf`",
            "- `reports/figures/intervention_component_reduction_student_t_stress_tail_2030.png` / `.pdf`",
        ]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "intervention_lever_scenarios.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    marginals, corr = load_inputs()
    rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for dependence in DEPENDENCE_STRUCTURES:
        for scenario in SCENARIOS:
            for year in sorted(marginals["year"].unique()):
                base = simulate_year(
                    marginals,
                    corr,
                    year=int(year),
                    dependence=dependence,
                    scenario=scenario,
                    rng=rng,
                )
                for lever in LEVER_SPECS:
                    sim = apply_lever(base, marginals, lever)
                    sim = add_risk_metrics(sim, marginals)
                    sim = add_loss(sim, marginals)
                    rows.append(summarize(sim))
                    component_rows.append(summarize_components(sim))

    summary = pd.DataFrame(rows)
    component_summary = pd.DataFrame(component_rows)
    reductions = add_reductions(summary)
    ranking = robust_ranking(reductions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "intervention_summary.csv", index=False)
    reductions.to_csv(OUT_DIR / "intervention_reductions.csv", index=False)
    ranking.to_csv(OUT_DIR / "intervention_robust_ranking.csv", index=False)
    component_summary.to_csv(OUT_DIR / "intervention_component_summary.csv", index=False)
    plot_mean_reduction(reductions)
    plot_cvar_reduction(reductions)
    plot_robust_ranking(ranking)
    plot_component_reduction(component_summary)
    write_report(summary, ranking, component_summary)
    print(f"Wrote intervention outputs to {OUT_DIR}")
    print(f"Wrote intervention report to {REPORT_DIR / 'intervention_lever_scenarios.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
