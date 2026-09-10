"""Run the joint Monte Carlo scenario simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import norm, t

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None


ROOT = Path(__file__).resolve().parents[2]
MARGINAL_PATH = ROOT / "data_processed" / "marginal_final" / "final_marginal_envelopes_2026_2030.csv"
CORR_PATH = ROOT / "data_processed" / "dependence" / "proposed_copula_correlation_matrix.csv"
OUT_DIR = ROOT / "data_processed" / "joint_monte_carlo"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

COPULA_VECTOR = [
    "bev_fcev_share_nsw",
    "hybrid_share_nsw",
    "petrol_share_nsw",
    "diesel_share_nsw",
    "renewable_share_nsw",
    "road_energy_use_national",
    "road_direct_emissions_national",
    "sydney_passenger_activity",
]

DEPENDENCE_STRUCTURES = ["independent", "gaussian", "student_t"]
SCENARIO_FAMILIES = ["conservative", "central", "optimistic", "stress_tail"]
DEPENDENCE_LABELS = {
    "independent": "Independent",
    "gaussian": "Gaussian",
    "student_t": "Student-t",
}
SCENARIO_LABELS = {
    "conservative": "Conservative",
    "central": "Central",
    "optimistic": "Optimistic",
    "stress_tail": "Stress tail",
}
BENEFICIAL = {"bev_fcev_share_nsw", "hybrid_share_nsw", "renewable_share_nsw"}
ADVERSE = {
    "petrol_share_nsw",
    "diesel_share_nsw",
    "road_energy_use_national",
    "road_direct_emissions_national",
    "sydney_passenger_activity",
}
N_SAMPLES = 20000
RNG_SEED = 20260620
T_DF = 4


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
            "savefig.dpi": 600,
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def nearest_positive_semidefinite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.clip(eigvals, epsilon, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    diag = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(diag, diag)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    marginals = pd.read_csv(MARGINAL_PATH)
    matrix = pd.read_csv(CORR_PATH, index_col="series_id")
    matrix = matrix.loc[COPULA_VECTOR, COPULA_VECTOR]
    matrix.iloc[:, :] = nearest_positive_semidefinite(matrix.to_numpy(dtype=float))
    return marginals, matrix


def sample_uniforms(
    rng: np.random.Generator,
    dependence: str,
    corr: pd.DataFrame,
    n_samples: int,
) -> np.ndarray:
    dim = len(COPULA_VECTOR)
    if dependence == "independent":
        return rng.random((n_samples, dim))

    corr_np = corr.to_numpy(dtype=float)
    if dependence == "gaussian":
        z = rng.multivariate_normal(np.zeros(dim), corr_np, size=n_samples)
        return np.clip(norm.cdf(z), 1e-6, 1.0 - 1e-6)

    if dependence == "student_t":
        z = rng.multivariate_normal(np.zeros(dim), corr_np, size=n_samples)
        scale = np.sqrt(rng.chisquare(T_DF, size=n_samples) / T_DF)[:, None]
        x = z / scale
        return np.clip(t.cdf(x, df=T_DF), 1e-6, 1.0 - 1e-6)

    raise ValueError(f"Unknown dependence structure: {dependence}")


def orient_quantiles(u: np.ndarray, scenario: str) -> np.ndarray:
    oriented = u.copy()
    for idx, series_id in enumerate(COPULA_VECTOR):
        if scenario == "central":
            continue
        if scenario == "conservative":
            if series_id in BENEFICIAL:
                oriented[:, idx] = 0.10 + 0.40 * u[:, idx]
            elif series_id in ADVERSE:
                oriented[:, idx] = 0.50 + 0.40 * u[:, idx]
        elif scenario == "optimistic":
            if series_id in BENEFICIAL:
                oriented[:, idx] = 0.50 + 0.40 * u[:, idx]
            elif series_id in ADVERSE:
                oriented[:, idx] = 0.10 + 0.40 * u[:, idx]
        elif scenario == "stress_tail":
            if series_id in BENEFICIAL:
                oriented[:, idx] = 0.02 + 0.28 * u[:, idx]
            elif series_id in ADVERSE:
                oriented[:, idx] = 0.70 + 0.28 * u[:, idx]
        else:
            raise ValueError(f"Unknown scenario family: {scenario}")
    return np.clip(oriented, 1e-6, 1.0 - 1e-6)


def inverse_marginal(u: np.ndarray, p10: float, p50: float, p90: float) -> np.ndarray:
    quantile_grid = np.array([0.0, 0.10, 0.50, 0.90, 1.0])
    value_grid = np.array([p10, p10, p50, p90, p90], dtype=float)
    return np.interp(u, quantile_grid, value_grid)


def simulate_year(
    marginals: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    year: int,
    dependence: str,
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    year_marg = marginals[marginals["year"] == year].set_index("series_id").loc[COPULA_VECTOR]
    u = orient_quantiles(sample_uniforms(rng, dependence, corr, N_SAMPLES), scenario)
    draws = {}
    for idx, series_id in enumerate(COPULA_VECTOR):
        row = year_marg.loc[series_id]
        draws[series_id] = inverse_marginal(
            u[:, idx],
            float(row["p10"]),
            float(row["p50"]),
            float(row["p90"]),
        )
    sim = pd.DataFrame(draws)
    sim["year"] = year
    sim["dependence"] = dependence
    sim["scenario_family"] = scenario
    return sim


def add_risk_metrics(sim: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    year = int(sim["year"].iloc[0])
    base = baseline[baseline["year"] == year].set_index("series_id")
    bev_ref = float(base.loc["bev_fcev_share_nsw", "p50"])
    renewable_ref = float(base.loc["renewable_share_nsw", "p50"])
    fossil_ref = float(base.loc["petrol_share_nsw", "p50"] + base.loc["diesel_share_nsw", "p50"])
    emissions_ref = float(base.loc["road_direct_emissions_national", "p50"])
    activity_ref = float(base.loc["sydney_passenger_activity", "p50"])

    sim = sim.copy()
    sim["low_emission_share_nsw"] = sim["bev_fcev_share_nsw"] + sim["hybrid_share_nsw"]
    sim["fossil_share_nsw"] = sim["petrol_share_nsw"] + sim["diesel_share_nsw"]
    sim["alignment_gap"] = sim["renewable_share_nsw"] - sim["bev_fcev_share_nsw"]
    sim["weak_alignment"] = (sim["bev_fcev_share_nsw"] >= bev_ref) & (sim["renewable_share_nsw"] < renewable_ref)
    sim["favorable_alignment"] = (sim["bev_fcev_share_nsw"] >= bev_ref) & (sim["renewable_share_nsw"] >= renewable_ref)
    sim["fossil_persistence"] = sim["fossil_share_nsw"] >= fossil_ref
    sim["emissions_pressure"] = sim["road_direct_emissions_national"] >= emissions_ref
    sim["activity_pressure"] = sim["sydney_passenger_activity"] >= activity_ref
    return sim


def summarize(sim: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "year": int(sim["year"].iloc[0]),
        "dependence": sim["dependence"].iloc[0],
        "scenario_family": sim["scenario_family"].iloc[0],
        "n_samples": len(sim),
        "weak_alignment_probability": float(sim["weak_alignment"].mean()),
        "favorable_alignment_probability": float(sim["favorable_alignment"].mean()),
        "fossil_persistence_probability": float(sim["fossil_persistence"].mean()),
        "emissions_pressure_probability": float(sim["emissions_pressure"].mean()),
        "activity_pressure_probability": float(sim["activity_pressure"].mean()),
    }
    for metric in [
        "bev_fcev_share_nsw",
        "hybrid_share_nsw",
        "renewable_share_nsw",
        "fossil_share_nsw",
        "road_direct_emissions_national",
        "alignment_gap",
    ]:
        values = sim[metric].to_numpy(dtype=float)
        row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
        row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
        row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
    return row


def plot_probability_summary(summary: pd.DataFrame) -> None:
    final = summary[(summary["year"] == 2030) & (summary["scenario_family"] == "central")].copy()
    final = final.sort_values(["dependence"])
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.bar(np.arange(len(final)), final["weak_alignment_probability"], color="tab:red", alpha=0.75)
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in final["dependence"]], rotation=25, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Weak alignment, central 2030")
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "joint_mc_weak_alignment_2030")


def plot_dependence_comparison(summary: pd.DataFrame) -> None:
    central = summary[(summary["year"] == 2030) & (summary["scenario_family"] == "central")].copy()
    central = central.sort_values("dependence")
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.bar(
        np.arange(len(central)),
        central["emissions_pressure_probability"],
        color="tab:blue",
        alpha=0.8,
    )
    ax.set_xticks(np.arange(len(central)))
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in central["dependence"]], rotation=25, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Emissions pressure, central 2030")
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "joint_mc_emissions_pressure_2030")


def plot_central_probability_panel(summary: pd.DataFrame) -> None:
    central = summary[(summary["year"] == 2030) & (summary["scenario_family"] == "central")].copy()
    central = central.sort_values("dependence")
    metrics = [
        ("weak_alignment_probability", "Weak alignment"),
        ("favorable_alignment_probability", "Favorable alignment"),
        ("fossil_persistence_probability", "Fossil persistence"),
        ("emissions_pressure_probability", "Emissions pressure"),
        ("activity_pressure_probability", "Activity pressure"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 2.625))
    colors = {"independent": "tab:gray", "gaussian": "tab:blue", "student_t": "tab:orange"}
    for offset, dependence in zip([-width, 0.0, width], DEPENDENCE_STRUCTURES):
        row = central[central["dependence"] == dependence].iloc[0]
        values = [float(row[col]) for col, _ in metrics]
        ax.bar(x + offset, values, width=width, label=DEPENDENCE_LABELS[dependence], color=colors[dependence], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("Probability")
    ax.set_title("Central 2030 probability summary")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=3, framealpha=0.85, handlelength=1.2, columnspacing=0.8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, "joint_mc_central_probability_panel_2030")


def plot_probability_trajectory(summary: pd.DataFrame, metric: str, title: str, stem: str) -> None:
    central = summary[summary["scenario_family"] == "central"].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    colors = {"independent": "tab:gray", "gaussian": "tab:blue", "student_t": "tab:orange"}
    for dependence in DEPENDENCE_STRUCTURES:
        sub = central[central["dependence"] == dependence].sort_values("year")
        ax.plot(
            sub["year"],
            sub[metric],
            marker="o",
            ms=2.5,
            color=colors[dependence],
            label=DEPENDENCE_LABELS[dependence],
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(central["year"].unique()))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=6)
    ax.legend(loc="best", framealpha=0.85, handlelength=1.3)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_scenario_heatmap(summary: pd.DataFrame, metric: str, title: str, stem: str) -> None:
    final = summary[summary["year"] == 2030].copy()
    pivot = final.pivot(index="scenario_family", columns="dependence", values=metric)
    pivot = pivot.loc[SCENARIO_FAMILIES, DEPENDENCE_STRUCTURES]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(DEPENDENCE_STRUCTURES)))
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in DEPENDENCE_STRUCTURES], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIO_FAMILIES)))
    ax.set_yticklabels([SCENARIO_LABELS[name] for name in SCENARIO_FAMILIES])
    for i in range(len(SCENARIO_FAMILIES)):
        for j in range(len(DEPENDENCE_STRUCTURES)):
            value = float(pivot.iloc[i, j])
            text_color = "white" if value < 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color=text_color)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_interval_panel(summary: pd.DataFrame, metric: str, label: str, unit: str, stem: str) -> None:
    central = summary[(summary["year"] == 2030) & (summary["scenario_family"] == "central")].copy()
    central = central.sort_values("dependence")
    x = np.arange(len(central))
    p05 = central[f"{metric}_p05"].astype(float).to_numpy()
    p50 = central[f"{metric}_p50"].astype(float).to_numpy()
    p95 = central[f"{metric}_p95"].astype(float).to_numpy()
    yerr = np.vstack([p50 - p05, p95 - p50])
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.errorbar(x, p50, yerr=yerr, fmt="o", color="tab:blue", capsize=3, ms=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DEPENDENCE_LABELS[x] for x in central["dependence"]], rotation=25, ha="right")
    ax.set_ylabel(unit)
    ax.set_title(label)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_low_vs_renewable(summary: pd.DataFrame) -> None:
    final = summary[summary["year"] == 2030].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    colors = {
        "conservative": "tab:red",
        "central": "tab:blue",
        "optimistic": "tab:green",
        "stress_tail": "tab:purple",
    }
    markers = {"independent": "o", "gaussian": "s", "student_t": "^"}
    offsets = {"independent": -0.015, "gaussian": 0.0, "student_t": 0.015}
    for _, row in final.iterrows():
        low_emission = float(row["bev_fcev_share_nsw_p50"]) + float(row["hybrid_share_nsw_p50"])
        renewable = float(row["renewable_share_nsw_p50"])
        ax.scatter(
            low_emission + offsets[row["dependence"]],
            renewable,
            s=16,
            marker=markers[row["dependence"]],
            color=colors[row["scenario_family"]],
            alpha=0.85,
        )
    scenario_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, markersize=4.5, label=SCENARIO_LABELS[scenario])
        for scenario, color in colors.items()
    ]
    dependence_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="none", markersize=4.5, label=DEPENDENCE_LABELS[dependence])
        for dependence, marker in markers.items()
    ]
    ax.set_xlabel("Low-emission share P50 (%)")
    ax.set_ylabel("Renewable share P50 (%)")
    ax.set_title("Uptake-electricity alignment, 2030")
    ax.tick_params(labelsize=6)
    first = ax.legend(handles=scenario_handles, loc="upper left", fontsize=4.7, framealpha=0.85, handlelength=1.0)
    ax.add_artist(first)
    ax.legend(handles=dependence_handles, loc="lower right", fontsize=4.7, framealpha=0.85, handlelength=1.0)
    fig.tight_layout()
    save_fig(fig, "joint_mc_low_emission_vs_renewable_2030")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(summary: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = summary[summary["year"] == 2030].sort_values(["scenario_family", "dependence"])
    cols = [
        "dependence",
        "scenario_family",
        "weak_alignment_probability",
        "favorable_alignment_probability",
        "fossil_persistence_probability",
        "emissions_pressure_probability",
    ]
    lines = [
        "# Joint Monte Carlo Scenario Simulation",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This step combines the final marginal envelopes with the dependence structures to generate joint 2026-2030 transition scenarios.",
        "",
        "## Inputs",
        "",
        "- `data_processed/marginal_final/final_marginal_envelopes_2026_2030.csv`",
        "- `data_processed/dependence/proposed_copula_correlation_matrix.csv`",
        "",
        "## Scenario Families",
        "",
        "- conservative: beneficial transition variables are sampled from lower quantiles and adverse variables from upper quantiles.",
        "- central: full marginal uncertainty is sampled without directional scenario bias.",
        "- optimistic: beneficial transition variables are sampled from upper quantiles and adverse variables from lower quantiles.",
        "- stress/tail: adverse-tail scenario with beneficial variables low and adverse variables high.",
        "",
        "## Dependence Structures",
        "",
        "- independent",
        "- Gaussian copula",
        "- Student-t copula with 4 degrees of freedom",
        "",
        "## 2030 Probability Summary",
        "",
    ]
    lines.extend(markdown_table(final[cols], cols))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Dependence changes joint event probabilities, especially when the same marginal envelopes are combined under Gaussian or Student-t co-movement.",
            "- Conservative, optimistic, and stress/tail families are directional scenario tests; saturated 0 or 1 probabilities should be read as scenario construction results, not calibrated event probabilities.",
            "- The stress/tail family is a diagnostic adverse-tail case, not a forecast claim.",
            "- The central/reference scenario is the main comparison for dependence effects.",
            "- These outputs provide the inputs for the CVaR, regret, and intervention analyses.",
            "",
            "## Outputs",
            "",
            "- `data_processed/joint_monte_carlo/joint_risk_summary.csv`",
            "- `data_processed/joint_monte_carlo/joint_sample_summary_2030.csv`",
            "- `reports/figures/joint_mc_weak_alignment_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_emissions_pressure_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_central_probability_panel_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_*_trajectory_2026_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_*_heatmap_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_*_interval_2030.png` / `.pdf`",
            "- `reports/figures/joint_mc_low_emission_vs_renewable_2030.png` / `.pdf`",
        ]
    )
    (REPORT_DIR / "joint_monte_carlo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    marginals, corr = load_inputs()
    rows: list[dict[str, object]] = []
    final_sample_rows: list[dict[str, object]] = []
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
                rows.append(summarize(sim))
                if int(year) == 2030:
                    final_sample_rows.append(
                        {
                            "dependence": dependence,
                            "scenario_family": scenario,
                            "bev_fcev_share_nsw_mean": float(sim["bev_fcev_share_nsw"].mean()),
                            "renewable_share_nsw_mean": float(sim["renewable_share_nsw"].mean()),
                            "fossil_share_nsw_mean": float(sim["fossil_share_nsw"].mean()),
                            "road_direct_emissions_national_mean": float(sim["road_direct_emissions_national"].mean()),
                        }
                    )

    summary = pd.DataFrame(rows)
    sample_summary = pd.DataFrame(final_sample_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "joint_risk_summary.csv", index=False)
    sample_summary.to_csv(OUT_DIR / "joint_sample_summary_2030.csv", index=False)
    plot_probability_summary(summary)
    plot_dependence_comparison(summary)
    plot_central_probability_panel(summary)
    plot_probability_trajectory(
        summary,
        "weak_alignment_probability",
        "Weak alignment, central scenario",
        "joint_mc_weak_alignment_trajectory_2026_2030",
    )
    plot_probability_trajectory(
        summary,
        "fossil_persistence_probability",
        "Fossil persistence, central scenario",
        "joint_mc_fossil_persistence_trajectory_2026_2030",
    )
    plot_probability_trajectory(
        summary,
        "favorable_alignment_probability",
        "Favorable alignment, central scenario",
        "joint_mc_favorable_alignment_trajectory_2026_2030",
    )
    plot_scenario_heatmap(
        summary,
        "weak_alignment_probability",
        "2030 weak-alignment probability",
        "joint_mc_weak_alignment_heatmap_2030",
    )
    plot_scenario_heatmap(
        summary,
        "fossil_persistence_probability",
        "2030 fossil-persistence probability",
        "joint_mc_fossil_persistence_heatmap_2030",
    )
    plot_scenario_heatmap(
        summary,
        "emissions_pressure_probability",
        "2030 emissions-pressure probability",
        "joint_mc_emissions_pressure_heatmap_2030",
    )
    plot_interval_panel(
        summary,
        "bev_fcev_share_nsw",
        "BEV/FCEV share, central 2030",
        "percent",
        "joint_mc_bev_fcev_interval_2030",
    )
    plot_interval_panel(
        summary,
        "renewable_share_nsw",
        "Renewable share, central 2030",
        "percent",
        "joint_mc_renewable_interval_2030",
    )
    plot_interval_panel(
        summary,
        "fossil_share_nsw",
        "Fossil share, central 2030",
        "percent",
        "joint_mc_fossil_share_interval_2030",
    )
    plot_interval_panel(
        summary,
        "road_direct_emissions_national",
        "Road emissions, central 2030",
        "Mt CO2-e",
        "joint_mc_road_emissions_interval_2030",
    )
    plot_interval_panel(
        summary,
        "alignment_gap",
        "Alignment gap, central 2030",
        "percentage points",
        "joint_mc_alignment_gap_interval_2030",
    )
    plot_low_vs_renewable(summary)
    write_report(summary)
    print(f"Wrote joint Monte Carlo outputs to {OUT_DIR}")
    print(f"Wrote joint Monte Carlo report to {REPORT_DIR / 'joint_monte_carlo.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
