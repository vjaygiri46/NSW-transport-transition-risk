"""Build the dependence assumptions and copula inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data_processed" / "model_inputs"
FINAL_DIR = ROOT / "data_processed" / "marginal_final"
OUT_DIR = ROOT / "data_processed" / "dependence"
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

ASSUMED_CORRELATIONS: dict[tuple[str, str], tuple[float, str]] = {
    ("bev_fcev_share_nsw", "hybrid_share_nsw"): (
        0.35,
        "Short overlap; both grow in recent data, but substitution risk keeps assumption moderate.",
    ),
    ("bev_fcev_share_nsw", "petrol_share_nsw"): (
        -0.55,
        "BEV/FCEV uptake should reduce petrol reliance, but fleet growth prevents one-for-one substitution.",
    ),
    ("bev_fcev_share_nsw", "diesel_share_nsw"): (
        -0.20,
        "Passenger electrification has weaker near-term effect on diesel persistence.",
    ),
    ("bev_fcev_share_nsw", "renewable_share_nsw"): (
        0.25,
        "Policy and infrastructure alignment can co-move, but direct historical evidence is short.",
    ),
    ("hybrid_share_nsw", "petrol_share_nsw"): (
        -0.35,
        "Hybrid uptake partially reduces petrol-only fleet share.",
    ),
    ("hybrid_share_nsw", "diesel_share_nsw"): (
        -0.15,
        "Hybrid uptake is less directly connected to diesel persistence.",
    ),
    ("petrol_share_nsw", "diesel_share_nsw"): (
        -0.45,
        "Fuel shares compete within fleet composition.",
    ),
    ("diesel_share_nsw", "sydney_passenger_activity"): (
        0.20,
        "Activity growth can preserve fossil-fuel demand pressure.",
    ),
    ("road_energy_use_national", "road_direct_emissions_national"): (
        0.70,
        "Energy use and direct emissions are strongly connected historically.",
    ),
    ("sydney_passenger_activity", "road_energy_use_national"): (
        0.40,
        "Activity growth is a demand-pressure proxy for road energy use.",
    ),
    ("sydney_passenger_activity", "road_direct_emissions_national"): (
        0.30,
        "Activity growth can pressure emissions, moderated by technology and fuel mix.",
    ),
    ("renewable_share_nsw", "road_direct_emissions_national"): (
        -0.20,
        "Cleaner electricity supports electrification benefits, but national road direct emissions are not electricity-only.",
    ),
}


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


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "fleet": pd.read_csv(INPUT_DIR / "fleet_broad_preferred_2010_2025.csv"),
        "low": pd.read_csv(INPUT_DIR / "low_emission_detail_2021_2025.csv"),
        "electricity": pd.read_csv(INPUT_DIR / "electricity_mix_nsw_aus.csv"),
        "transport": pd.read_csv(INPUT_DIR / "transport_calibration_national.csv"),
        "sydney": pd.read_csv(INPUT_DIR / "sydney_activity_proxy.csv"),
        "final": pd.read_csv(FINAL_DIR / "final_marginal_envelopes_2026_2030.csv"),
    }


def road_topic_series(transport: pd.DataFrame, topic: str, variable: str, series_id: str) -> pd.DataFrame:
    sub = transport[(transport["topic"] == topic) & (transport["variable"] == variable)].copy()
    sub["year"] = sub["period"].str.slice(0, 4).astype(int)
    sub[series_id] = sub["value"].astype(float)
    return sub[["year", series_id]]


def historical_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fleet = data["fleet"][data["fleet"]["jurisdiction"] == "NSW"].copy()
    fleet = fleet[
        ["year", "petrol_share_percent", "diesel_share_percent"]
    ].rename(
        columns={
            "petrol_share_percent": "petrol_share_nsw",
            "diesel_share_percent": "diesel_share_nsw",
        }
    )

    low = data["low"][data["low"]["jurisdiction"] == "NSW"].copy()
    low = low[["year", "bev_fcev_share_percent", "hybrid_share_percent"]].rename(
        columns={
            "bev_fcev_share_percent": "bev_fcev_share_nsw",
            "hybrid_share_percent": "hybrid_share_nsw",
        }
    )

    electricity = data["electricity"][
        (data["electricity"]["jurisdiction"] == "NSW")
        & (data["electricity"]["period_type"] == "calendar_year")
    ].copy()
    electricity["year"] = electricity["period"].astype(int)
    electricity = electricity[["year", "renewable_share_percent"]].rename(
        columns={"renewable_share_percent": "renewable_share_nsw"}
    )

    sydney = data["sydney"][data["sydney"]["mode_slug"] == "total"].copy()
    sydney["year"] = sydney["period"].str.slice(0, 4).astype(int)
    sydney = sydney[["year", "value"]].rename(columns={"value": "sydney_passenger_activity"})

    transport_energy = road_topic_series(
        data["transport"], "road_energy_use_by_vehicle_type", "Total road", "road_energy_use_national"
    )
    transport_emissions = road_topic_series(
        data["transport"], "road_direct_ghg_by_vehicle_type", "Total road", "road_direct_emissions_national"
    )

    panel = fleet
    for frame in [low, electricity, sydney, transport_energy, transport_emissions]:
        panel = panel.merge(frame, on="year", how="outer")
    return panel.sort_values("year")


def historical_correlations(panel: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, left in enumerate(COPULA_VECTOR):
        for right in COPULA_VECTOR[i + 1 :]:
            pair = panel[["year", left, right]].dropna()
            if len(pair) >= 5:
                corr = float(pair[left].corr(pair[right], method="spearman"))
                source = "historical_overlap"
                note = f"{len(pair)} overlapping annual observations."
            else:
                corr = ""
                source = "insufficient_overlap"
                note = f"{len(pair)} overlapping annual observations; do not estimate directly."
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_overlap": len(pair),
                    "spearman_rho": corr,
                    "source": source,
                    "note": note,
                }
            )
    return rows


def assumed_corr(left: str, right: str) -> tuple[float, str]:
    if (left, right) in ASSUMED_CORRELATIONS:
        return ASSUMED_CORRELATIONS[(left, right)]
    if (right, left) in ASSUMED_CORRELATIONS:
        return ASSUMED_CORRELATIONS[(right, left)]
    return 0.0, "Set to zero in first-pass compact copula because no specific dependence hypothesis is documented."


def nearest_positive_semidefinite(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.clip(eigvals, epsilon, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    diag = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(diag, diag)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def proposed_matrix() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    n = len(COPULA_VECTOR)
    matrix = np.eye(n)
    assumption_rows: list[dict[str, object]] = []
    for i, left in enumerate(COPULA_VECTOR):
        for j, right in enumerate(COPULA_VECTOR):
            if j <= i:
                continue
            value, rationale = assumed_corr(left, right)
            matrix[i, j] = value
            matrix[j, i] = value
            assumption_rows.append(
                {
                    "left": left,
                    "right": right,
                    "proposed_rho": value,
                    "rationale": rationale,
                }
            )
    repaired = nearest_positive_semidefinite(matrix)
    repaired[np.abs(repaired) < 1e-12] = 0.0
    matrix_df = pd.DataFrame(repaired, index=COPULA_VECTOR, columns=COPULA_VECTOR)
    return matrix_df, assumption_rows


def write_matrix_csv(path: Path, matrix: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=True, index_label="series_id")


def plot_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.25, 4.2))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    labels = [name.replace("_", "\n") for name in matrix.index]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticklabels(labels, fontsize=5.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.8)
    ax.set_title("Proposed Gaussian / Student-t copula correlation")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "dependence_correlation_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "dependence_correlation_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    historical_rows: list[dict[str, object]],
    assumption_rows: list[dict[str, object]],
    matrix: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    focused_assumptions = [row for row in assumption_rows if abs(float(row["proposed_rho"])) > 0]
    historical_key = [
        row
        for row in historical_rows
        if row["source"] == "historical_overlap"
        and int(row["n_overlap"]) >= 10
        and abs(float(row["spearman_rho"])) >= 0.5
    ]
    lines = [
        "# Dependence Model",
        "",
        "Status date: 2026-06-20",
        "",
        "## Purpose",
        "",
        "This step defines dependence structures for the joint simulation. It does not run the Monte Carlo risk calculations.",
        "",
        "## Copula Vector",
        "",
    ]
    lines.extend([f"{idx}. `{name}`" for idx, name in enumerate(COPULA_VECTOR, start=1)])
    lines.extend(
        [
            "",
            "## Structures To Test",
            "",
            "1. Independent sampling baseline.",
            "2. Gaussian copula using the proposed rank-correlation matrix.",
            "3. Student-t copula using the same matrix with low degrees of freedom for tail-dependence stress testing.",
            "",
            "Optional Archimedean copulas are deferred unless a directional tail-dependence hypothesis becomes defensible.",
            "",
            "## Historical Rank Correlations",
            "",
            "Historical correlations are used only where overlapping annual observations are meaningful. Short BEV/FCEV and hybrid overlaps are kept in the CSV diagnostics but excluded from this highlighted table.",
            "",
            "High correlations in long trending annual series are diagnostic evidence, not causal proof. The proposed copula matrix therefore combines historical diagnostics with conservative documented assumptions.",
            "",
        ]
    )
    if historical_key:
        lines.extend(markdown_table(historical_key, ["left", "right", "n_overlap", "spearman_rho", "note"]))
    else:
        lines.append("No strong historical overlap correlations met the reporting threshold.")
    lines.extend(
        [
            "",
            "## Proposed Nonzero Dependence Assumptions",
            "",
        ]
    )
    lines.extend(markdown_table(focused_assumptions, ["left", "right", "proposed_rho", "rationale"]))
    lines.extend(
        [
            "",
            "## Copula Matrix",
            "",
            "The proposed matrix is repaired to the nearest positive-semidefinite correlation matrix before use.",
            "",
            "| series_id | " + " | ".join(COPULA_VECTOR) + " |",
            "|" + "|".join(["---"] * (len(COPULA_VECTOR) + 1)) + "|",
        ]
    )
    for series_id, row in matrix.iterrows():
        values = [f"{float(row[col]):.3g}" for col in COPULA_VECTOR]
        lines.append("| " + " | ".join([series_id, *values]) + " |")
    lines.extend(
        [
            "",
            "## Joint simulation inputs",
            "",
            "- Use `data_processed/marginal_final/final_marginal_envelopes_2026_2030.csv` as marginal inputs.",
            "- Use `data_processed/dependence/proposed_copula_correlation_matrix.csv` for Gaussian and Student-t copulas.",
            "- Test independence, Gaussian copula, and Student-t copula in the joint Monte Carlo simulation.",
            "",
            "## Outputs",
            "",
            "- `data_processed/dependence/historical_rank_correlations.csv`",
            "- `data_processed/dependence/dependence_assumptions.csv`",
            "- `data_processed/dependence/proposed_copula_correlation_matrix.csv`",
            "- `reports/figures/dependence_correlation_matrix.png` / `.pdf`",
        ]
    )
    (REPORT_DIR / "dependence_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    apply_style()
    data = load_inputs()
    panel = historical_panel(data)
    historical_rows = historical_correlations(panel)
    matrix, assumption_rows = proposed_matrix()
    write_csv(
        OUT_DIR / "historical_rank_correlations.csv",
        historical_rows,
        ["left", "right", "n_overlap", "spearman_rho", "source", "note"],
    )
    write_csv(
        OUT_DIR / "dependence_assumptions.csv",
        assumption_rows,
        ["left", "right", "proposed_rho", "rationale"],
    )
    write_matrix_csv(OUT_DIR / "proposed_copula_correlation_matrix.csv", matrix)
    plot_heatmap(matrix)
    write_report(historical_rows, assumption_rows, matrix)
    print(f"Wrote dependence outputs to {OUT_DIR}")
    print(f"Wrote dependence report to {REPORT_DIR / 'dependence_model.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
