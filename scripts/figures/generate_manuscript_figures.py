from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
OUT = ROOT / "figures"
SUPP_OUT = OUT / "supplementary"

SERIES_LABELS = {
    "fleet_total_nsw": "NSW registered vehicles",
    "bev_fcev_share_nsw": "BEV/FCEV share",
    "hybrid_share_nsw": "Hybrid share",
    "petrol_share_nsw": "Petrol share",
    "diesel_share_nsw": "Diesel share",
    "renewable_share_nsw": "Renewable share",
    "road_direct_emissions_national": "Road emissions",
    "road_energy_use_national": "Road energy use",
    "sydney_passenger_activity": "Sydney passenger activity",
}

DEPENDENCE_LABELS = {
    "independent": "Independent",
    "gaussian": "Gaussian",
    "student_t": r"Student-$t$",
}

SCENARIO_LABELS = {
    "central": "Central",
    "conservative": "Conservative",
    "optimistic": "Optimistic",
    "stress_tail": "Stress-tail",
}

LEVER_LABELS = {
    "none": "No shift",
    "uptake_push": "Vehicle-uptake shift",
    "renewable_push": "Renewable-alignment shift",
    "fossil_reduction": "Fossil-fleet reduction shift",
    "balanced_package": "Coordinated transition shift",
    "delay_case": "Delayed-transition comparison",
}

COMPONENT_LABELS = {
    "weak_alignment_probability": "Weak alignment",
    "low_uptake_shortfall": "Low uptake",
    "renewable_shortfall": "Renewable shortfall",
    "fossil_excess": "Fossil excess",
    "emissions_excess": "Emissions excess",
    "activity_excess": "Activity excess",
}


def apply_style() -> None:
    if scienceplots is not None:
        try:
            plt.style.use(["science"])
        except Exception:
            plt.style.use("default")
    else:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.15,
            # Render text with LaTeX in a serif (Computer Modern) face so that
            # figures match the elsarticle manuscript typography.
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def save_supp(fig: plt.Figure, stem: str) -> None:
    SUPP_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SUPP_OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def fig02_marginal_panel() -> None:
    df = pd.read_csv(DATA / "marginal_final" / "final_marginal_envelopes_2026_2030.csv")
    series = [
        "bev_fcev_share_nsw",
        "hybrid_share_nsw",
        "petrol_share_nsw",
        "diesel_share_nsw",
        "renewable_share_nsw",
        "road_direct_emissions_national",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True)
    axes = axes.ravel()
    labels = list("ABCDEF")
    for ax, sid, panel in zip(axes, series, labels):
        sub = df[df["series_id"] == sid].sort_values("year")
        years = sub["year"].to_numpy()
        p10 = sub["p10"].to_numpy()
        p50 = sub["p50"].to_numpy()
        p90 = sub["p90"].to_numpy()
        ax.fill_between(years, p10, p90, color="#9ECAE1", alpha=0.45, linewidth=0)
        ax.plot(years, p50, color="#08519C", marker="o", markersize=3.0, label=r"P$_{50}$")
        ax.plot(years, p10, color="#6BAED6", linewidth=0.75)
        ax.plot(years, p90, color="#6BAED6", linewidth=0.75)
        ax.set_title(SERIES_LABELS[sid])
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=0)
        unit = sub["unit"].iloc[0]
        ax.set_ylabel(r"Mt CO$_2$-e" if unit == "Mt CO2-e" else unit)
        add_panel_label(ax, panel)
    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    save(fig, "fig02_marginal_envelope_panel")


def fig03_dependence_matrix() -> None:
    df = pd.read_csv(DATA / "dependence" / "proposed_copula_correlation_matrix.csv")
    labels = df["series_id"].map(SERIES_LABELS).fillna(df["series_id"]).tolist()
    mat = df.drop(columns=["series_id"]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    save(fig, "fig03_dependence_matrix")


def fig04_central_probabilities() -> None:
    df = pd.read_csv(DATA / "joint_monte_carlo" / "joint_risk_summary.csv")
    sub = df[(df["year"] == 2030) & (df["scenario_family"] == "central")].copy()
    sub["dependence_label"] = sub["dependence"].map(DEPENDENCE_LABELS)
    metrics = [
        ("weak_alignment_probability", "Weak\nalignment"),
        ("favorable_alignment_probability", "Favourable\nalignment"),
        ("fossil_persistence_probability", "Fossil\npersistence"),
        ("emissions_pressure_probability", "Emissions\npressure"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, dep in enumerate(["independent", "gaussian", "student_t"]):
        row = sub[sub["dependence"] == dep].iloc[0]
        vals = [row[m[0]] for m in metrics]
        ax.bar(x + (idx - 1) * width, vals, width, label=DEPENDENCE_LABELS[dep], color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig04_central_joint_probabilities")


def heatmap(ax: plt.Axes, data: pd.DataFrame, value_col: str, title: str, vmin: float, vmax: float) -> None:
    deps = ["independent", "gaussian", "student_t"]
    scenarios = ["central", "conservative", "optimistic", "stress_tail"]
    pivot = data.pivot(index="scenario_family", columns="dependence", values=value_col)
    mat = pivot.loc[scenarios, deps].to_numpy()
    im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(deps)))
    ax.set_xticklabels([DEPENDENCE_LABELS[d] for d in deps], rotation=25, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    return im


def fig05_downside_risk() -> None:
    df = pd.read_csv(DATA / "risk_robustness" / "robustness_summary.csv")
    sub = df[
        (df["year"] == 2030)
        & (df["weight_scheme"] == "baseline")
        & (df["threshold_mode"] == "baseline")
        & (np.isclose(df["cvar_level"], 0.95))
    ].copy()
    vmax = max(sub["mean_loss"].max(), sub["cvar_loss"].max())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.25))
    im0 = heatmap(axes[0], sub, "mean_loss", "Mean loss", 0, vmax)
    im1 = heatmap(axes[1], sub, "cvar_loss", r"CVaR$_{95}$ loss", 0, vmax)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.28, wspace=0.32)
    cbar = fig.colorbar(
        im1,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.08,
        pad=0.20,
    )
    cbar.set_label("Transition-loss index")
    save(fig, "fig05_downside_risk_panel")


def fig06_robustness() -> None:
    df = pd.read_csv(DATA / "risk_robustness" / "scenario_stability.csv")
    counts = (
        df["top_cvar_loss_scenario"]
        .value_counts()
        .reindex(["central", "conservative", "optimistic", "stress_tail"], fill_value=0)
    )
    labels = [SCENARIO_LABELS[s] for s in counts.index]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518", "#BDBDBD", "#E45756"])
    ax.set_ylabel("Count across sensitivity variants")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save(fig, "fig06_robustness_stability")


def fig07_driver_shift_ranking() -> None:
    df = pd.read_csv(DATA / "intervention_levers" / "intervention_robust_ranking.csv")
    order = [
        "balanced_package",
        "fossil_reduction",
        "renewable_push",
        "uptake_push",
        "delay_case",
    ]
    sub = df.set_index("lever").loc[order].reset_index()
    sub = sub.iloc[::-1].copy()
    labels = [LEVER_LABELS[x] for x in sub["lever"]]
    y = np.arange(len(sub))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    mean = sub["mean_reduction_median"].to_numpy()
    cvar = sub["cvar95_reduction_median"].to_numpy()
    mean_min = sub["mean_reduction_min"].to_numpy()
    cvar_min = sub["cvar95_reduction_min"].to_numpy()
    ax.barh(y - height / 2, mean, height, color="#4C78A8", label="Median mean-loss reduction")
    ax.barh(y + height / 2, cvar, height, color="#F58518", label=r"Median CVaR$_{95}$ reduction")
    ax.hlines(y - height / 2, mean_min, mean, color="#1F3349", linewidth=0.8)
    ax.hlines(y + height / 2, cvar_min, cvar, color="#6B3D09", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Reduction versus no shift")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout()
    save(fig, "fig07_driver_shift_ranking")


def fiscal_period_to_year(period: str) -> int:
    text = str(period)
    if "-" in text:
        return int(text.split("-")[0])
    return int(float(text))


def observed_series() -> dict[str, pd.DataFrame]:
    fleet = pd.read_csv(DATA / "model_inputs" / "fleet_broad_preferred_2010_2025.csv")
    low = pd.read_csv(DATA / "model_inputs" / "low_emission_detail_2021_2025.csv")
    elec = pd.read_csv(DATA / "model_inputs" / "electricity_mix_nsw_aus.csv")
    transport = pd.read_csv(DATA / "model_inputs" / "transport_calibration_national.csv")
    activity = pd.read_csv(DATA / "model_inputs" / "sydney_activity_proxy.csv")

    out: dict[str, pd.DataFrame] = {}

    def make(df: pd.DataFrame, year_col: str, value_col: str, unit: str) -> pd.DataFrame:
        return (
            df[[year_col, value_col]]
            .rename(columns={year_col: "year", value_col: "value"})
            .assign(unit=unit)
            .dropna()
            .sort_values("year")
        )

    nsw_fleet = fleet[fleet["jurisdiction"] == "NSW"].copy()
    out["fleet_total_nsw"] = make(nsw_fleet, "year", "total_vehicles", "vehicles")
    out["petrol_share_nsw"] = make(nsw_fleet, "year", "petrol_share_percent", "percent")
    out["diesel_share_nsw"] = make(nsw_fleet, "year", "diesel_share_percent", "percent")

    nsw_low = low[low["jurisdiction"] == "NSW"].copy()
    out["bev_fcev_share_nsw"] = make(nsw_low, "year", "bev_fcev_share_percent", "percent")
    out["hybrid_share_nsw"] = make(nsw_low, "year", "hybrid_share_percent", "percent")

    nsw_elec = elec[(elec["jurisdiction"] == "NSW") & (elec["period_type"] == "financial_year")].copy()
    nsw_elec["year"] = nsw_elec["period"].map(fiscal_period_to_year)
    out["renewable_share_nsw"] = make(nsw_elec, "year", "renewable_share_percent", "percent")

    road_em = transport[
        (transport["variable"] == "Total road") & (transport["measure"] == "direct_ghg_emissions")
    ].copy()
    road_em["year"] = road_em["period"].map(fiscal_period_to_year)
    out["road_direct_emissions_national"] = make(road_em, "year", "value", "Mt CO2-e")

    road_energy = transport[
        (transport["variable"] == "Total road") & (transport["measure"] == "energy_use")
    ].copy()
    road_energy["year"] = road_energy["period"].map(fiscal_period_to_year)
    out["road_energy_use_national"] = make(road_energy, "year", "value", "PJ")

    total_activity = activity[activity["mode_slug"] == "total"].copy()
    total_activity["year"] = total_activity["period"].map(fiscal_period_to_year)
    out["sydney_passenger_activity"] = make(
        total_activity, "year", "value", "billion passenger-km"
    )

    return out


def supp01_baseline_context() -> None:
    series = observed_series()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.1))
    axes = axes.ravel()
    labels = list("ABCD")

    fleet = series["petrol_share_nsw"].merge(
        series["diesel_share_nsw"], on="year", suffixes=("_petrol", "_diesel")
    )
    axes[0].plot(fleet["year"], fleet["value_petrol"], label="Petrol", color="#4C78A8")
    axes[0].plot(fleet["year"], fleet["value_diesel"], label="Diesel", color="#F58518")
    axes[0].set_title("NSW fossil-fuel fleet shares")
    axes[0].set_ylabel("percent")
    axes[0].legend(frameon=False, ncol=1, loc="best")

    low = series["bev_fcev_share_nsw"].merge(
        series["hybrid_share_nsw"], on="year", suffixes=("_bev", "_hybrid")
    )
    axes[1].plot(low["year"], low["value_bev"], marker="o", label="BEV/FCEV", color="#54A24B")
    axes[1].plot(low["year"], low["value_hybrid"], marker="o", label="Hybrid", color="#E45756")
    axes[1].set_title("NSW low-emission vehicle shares")
    axes[1].set_ylabel("percent")
    axes[1].legend(frameon=False, ncol=1, loc="best")

    renew = series["renewable_share_nsw"]
    axes[2].plot(renew["year"], renew["value"], marker="o", color="#72B7B2")
    axes[2].set_title("NSW renewable electricity share")
    axes[2].set_ylabel("percent")
    axes[2].set_xlabel("Year")

    emissions = series["road_direct_emissions_national"]
    energy = series["road_energy_use_national"]
    ax = axes[3]
    ax.plot(emissions["year"], emissions["value"], color="#B279A2", label="Emissions")
    ax.set_ylabel(r"Mt CO$_2$-e")
    ax2 = ax.twinx()
    ax2.plot(energy["year"], energy["value"], color="#FF9DA6", label="Energy use")
    ax2.set_ylabel("PJ")
    ax.set_title("National road transport pressure")
    ax.set_xlabel("Year")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    for ax, panel in zip(axes, labels):
        add_panel_label(ax, panel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, prune="both"))
    fig.tight_layout()
    save_supp(fig, "supp01_observed_context")


def supp02_model_forecast_comparison() -> None:
    observed = observed_series()
    forecasts = pd.read_csv(DATA / "marginal_benchmark" / "forecast_comparison_2026_2030.csv")
    envelopes = pd.read_csv(DATA / "marginal_final" / "final_marginal_envelopes_2026_2030.csv")
    selected = pd.read_csv(DATA / "marginal_benchmark" / "selected_marginal_treatments.csv")
    selected_map = dict(zip(selected["series_id"], selected["selected_method"]))
    series = [
        "fleet_total_nsw",
        "petrol_share_nsw",
        "diesel_share_nsw",
        "bev_fcev_share_nsw",
        "hybrid_share_nsw",
        "renewable_share_nsw",
        "road_energy_use_national",
        "road_direct_emissions_national",
        "sydney_passenger_activity",
    ]
    method_order = ["linear_trend", "holt_linear", "arima_drift"]
    method_labels = {
        "linear_trend": "Linear trend",
        "holt_linear": "Holt linear",
        "arima_drift": "ARIMA drift",
    }
    colors = {"linear_trend": "#F58518", "holt_linear": "#54A24B", "arima_drift": "#B279A2"}
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.4), sharex=False)
    axes = axes.ravel()
    for ax, sid, panel in zip(axes, series, list("ABCDEFGHI")):
        obs = observed[sid]
        env = envelopes[envelopes["series_id"] == sid].sort_values("year")
        ax.plot(obs["year"], obs["value"], color="black", marker="o", markersize=2.4, label="Observed")
        ax.fill_between(
            env["year"].to_numpy(),
            env["p10"].to_numpy(),
            env["p90"].to_numpy(),
            color="#9ECAE1",
            alpha=0.35,
            linewidth=0,
            label=r"Final P$_{10}$--P$_{90}$",
        )
        ax.plot(env["year"], env["p50"], color="#08519C", linewidth=1.2, label=r"Final P$_{50}$")
        for method in method_order:
            f = forecasts[(forecasts["series_id"] == sid) & (forecasts["method"] == method)]
            if f.empty:
                continue
            lw = 1.3 if selected_map.get(sid) == method else 0.85
            ls = "-" if selected_map.get(sid) == method else "--"
            ax.plot(
                f["year"],
                f["forecast"],
                color=colors[method],
                linestyle=ls,
                linewidth=lw,
                label=method_labels[method],
            )
        ax.axvline(2025.5, color="#777777", linewidth=0.7, linestyle=":")
        ax.set_title(SERIES_LABELS[sid])
        u = env["unit"].iloc[0]
        ax.set_ylabel(r"Mt CO$_2$-e" if u == "Mt CO2-e" else u)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, prune="both"))
        add_panel_label(ax, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.0, h_pad=1.1)
    save_supp(fig, "supp02_model_forecasts_with_final_envelopes")


def supp03_model_error_summary() -> None:
    df = pd.read_csv(DATA / "marginal_benchmark" / "benchmark_metrics.csv")
    keep = ["linear_trend", "holt_linear", "arima_drift"]
    sub = df[df["method"].isin(keep)].copy()
    series = [
        "fleet_total_nsw",
        "petrol_share_nsw",
        "diesel_share_nsw",
        "renewable_share_nsw",
        "road_energy_use_national",
        "road_direct_emissions_national",
        "sydney_passenger_activity",
    ]
    pivot = sub.pivot(index="series_id", columns="method", values="mape_percent").reindex(series)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([SERIES_LABELS[s] for s in series])
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(["Linear", "Holt", "ARIMA drift"], rotation=20, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if math.isnan(val):
                text = "NA"
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"Rolling MAPE (\%)")
    fig.tight_layout()
    save_supp(fig, "supp03_marginal_benchmark_mape")


def supp04_joint_probability_detail() -> None:
    df = pd.read_csv(DATA / "joint_monte_carlo" / "joint_risk_summary.csv")
    metrics = [
        ("weak_alignment_probability", "Weak alignment"),
        ("favorable_alignment_probability", "Favourable alignment"),
        ("fossil_persistence_probability", "Fossil persistence"),
        ("emissions_pressure_probability", "Emissions pressure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True, sharey=True)
    for ax, (metric, title), panel in zip(axes.ravel(), metrics, list("ABCD")):
        for dep, color in zip(["independent", "gaussian", "student_t"], ["#4C78A8", "#F58518", "#54A24B"]):
            sub = df[(df["scenario_family"] == "central") & (df["dependence"] == dep)].sort_values("year")
            ax.plot(sub["year"], sub[metric], marker="o", markersize=2.8, color=color, label=DEPENDENCE_LABELS[dep])
        ax.set_title(title)
        ax.set_xticks([2026, 2027, 2028, 2029, 2030])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        add_panel_label(ax, panel)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    save_supp(fig, "supp04_central_joint_probability_trajectories")


def supp05_risk_and_intervention_detail() -> None:
    risk = pd.read_csv(DATA / "risk_decision" / "risk_metric_summary.csv")
    risk2030 = risk[risk["year"] == 2030].copy()
    comps = pd.read_csv(DATA / "risk_decision" / "loss_component_summary.csv")
    comp2030 = comps[
        (comps["year"] == 2030)
        & (comps["dependence"] == "student_t")
    ].copy()
    reductions = pd.read_csv(DATA / "intervention_levers" / "intervention_reductions.csv")
    red2030 = reductions[
        (reductions["year"] == 2030)
        & (reductions["scenario_family"] == "stress_tail")
    ].copy()

    fig = plt.figure(figsize=(7.8, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.45])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    ]
    deps = ["independent", "gaussian", "student_t"]
    scenarios = ["central", "conservative", "optimistic", "stress_tail"]
    pivot = risk2030.pivot(index="scenario_family", columns="dependence", values="cvar90_loss")
    mat = pivot.loc[scenarios, deps].to_numpy()
    im = axes[0].imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(0.45, float(np.nanmax(mat))), aspect="auto")
    axes[0].set_xticks(range(len(deps)))
    axes[0].set_xticklabels([DEPENDENCE_LABELS[d] for d in deps], rotation=25, ha="right")
    axes[0].set_yticks(range(len(scenarios)))
    axes[0].set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_title(r"CVaR$_{90}$ loss")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[0].text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    components = [
        "weak_alignment",
        "low_uptake_shortfall",
        "renewable_shortfall",
        "fossil_excess",
        "emissions_excess",
        "activity_excess",
    ]
    available_components = [c for c in components if c in comp2030.columns]
    if not available_components:
        available_components = [c for c in COMPONENT_LABELS if c in comp2030.columns]
    comp_pivot = comp2030.set_index("scenario_family")[available_components].reindex(scenarios)
    comp_mat = comp_pivot.to_numpy(dtype=float)
    im_comp = axes[1].imshow(comp_mat, cmap="YlGnBu", vmin=0, vmax=max(0.65, float(np.nanmax(comp_mat))), aspect="auto")
    axes[1].set_xticks(range(len(available_components)))
    axes[1].set_xticklabels(
        [COMPONENT_LABELS.get(c, c.replace("_", " ")) for c in available_components],
        rotation=35,
        ha="right",
    )
    axes[1].set_yticks(range(len(scenarios)))
    axes[1].set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_title(r"Student-$t$ component values")
    for i in range(comp_mat.shape[0]):
        for j in range(comp_mat.shape[1]):
            axes[1].text(j, i, f"{comp_mat[i, j]:.2f}", ha="center", va="center", fontsize=6.3)
    fig.colorbar(im_comp, ax=axes[1], fraction=0.046, pad=0.04)

    order = ["balanced_package", "fossil_reduction", "renewable_push", "uptake_push", "delay_case"]
    agg = red2030.groupby("lever", as_index=False)[["mean_loss_reduction_vs_none", "cvar95_reduction_vs_none"]].median()
    agg = agg.set_index("lever").loc[order].iloc[::-1]
    y = np.arange(len(agg))
    axes[2].barh(y - 0.18, agg["mean_loss_reduction_vs_none"], 0.36, label="Mean", color="#4C78A8")
    axes[2].barh(y + 0.18, agg["cvar95_reduction_vs_none"], 0.36, label=r"CVaR$_{95}$", color="#F58518")
    axes[2].axvline(0, color="black", linewidth=0.7)
    axes[2].set_yticks(y)
    short_lever_labels = {
        "balanced_package": "Coordinated",
        "fossil_reduction": "Fossil-fleet",
        "renewable_push": "Renewable",
        "uptake_push": "Vehicle uptake",
        "delay_case": "Delayed",
    }
    axes[2].set_yticklabels([short_lever_labels[l] for l in agg.index])
    axes[2].set_title("Stress-tail reductions")
    axes[2].set_xlabel("Median reduction")
    axes[2].legend(frameon=False, ncol=1, loc="lower right")

    for ax, panel in zip(axes, list("ABC")):
        add_panel_label(ax, panel)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save_supp(fig, "supp05_risk_and_intervention_detail")


def fig08_driver_shift_components() -> None:
    df = pd.read_csv(DATA / "intervention_levers" / "intervention_component_summary.csv")
    sub = df[
        (df["year"] == 2030)
        & (df["dependence"] == "student_t")
        & (df["scenario_family"] == "stress_tail")
    ].copy()
    order = [
        "none",
        "balanced_package",
        "renewable_push",
        "fossil_reduction",
        "uptake_push",
        "delay_case",
    ]
    components = list(COMPONENT_LABELS)
    mat = sub.set_index("lever").loc[order, components].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.65, float(np.nanmax(mat))))
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([COMPONENT_LABELS[c] for c in components], rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LEVER_LABELS[l] for l in order])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Component value")
    fig.tight_layout()
    save(fig, "fig08_driver_shift_components")


def main() -> None:
    apply_style()
    fig02_marginal_panel()
    fig03_dependence_matrix()
    fig04_central_probabilities()
    fig05_downside_risk()
    fig06_robustness()
    fig07_driver_shift_ranking()
    fig08_driver_shift_components()
    supp01_baseline_context()
    supp02_model_forecast_comparison()
    supp03_model_error_summary()
    supp04_joint_probability_detail()
    supp05_risk_and_intervention_detail()


if __name__ == "__main__":
    main()
