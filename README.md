# Joint probabilistic transition risk in New South Wales road transport (2026–2030)

Code and processed data for the manuscript *"Joint probabilistic transition risk across
fleet turnover, renewable electricity, and road emissions in New South Wales,
2026–2030"* (submitted to *Case Studies on Transport Policy*).

The analysis converts uneven annual public-sector indicators (vehicle-fleet
composition, renewable electricity share, road energy use, road emissions, and
passenger activity) into marginal forecast envelopes, links them through
independent, Gaussian-copula, and Student-*t*-copula dependence assumptions,
propagates them through Monte Carlo simulation, and summarises the joint outcomes
with event probabilities, a composite transition-loss index, and conditional
value-at-risk (CVaR) downside-risk metrics, plus robustness checks and bounded
driver-shift scenarios.

## Repository layout

```
.
├── data_raw/            Instructions for obtaining the original workbooks
├── data_processed/      Cleaned, harmonised CSVs produced by the pipeline
├── data_sources/        Source registry and workbook inventory
├── scripts/
│   ├── ingest/          Download official source files
│   ├── clean/           Parse raw workbooks into tidy CSVs
│   ├── build/           Assemble harmonised model inputs
│   ├── modeling/        Marginals, dependence, Monte Carlo, risk, robustness, levers
│   └── figures/         generate_manuscript_figures.py (publication figures)
├── figures/             Reproduced manuscript figures and supplementary panels
├── requirements.txt
└── README.md
```

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Figure generation renders text with LaTeX (`text.usetex=True`), so a LaTeX
distribution (TeX Live or MiKTeX) with `latex` and `dvipng` on the PATH is required
to reproduce the figures. The processed data in `data_processed/` is included, so the
figures and results can be reproduced without re-running the full ingestion/cleaning
chain.

## Reproducing the results

Run from the repository root. Steps 1–2 are optional if you use the bundled
`data_processed/`. Rebuilding those processed files requires the original releases
listed in `data_sources/source_registry.csv`. Some archived files must be downloaded
or extracted manually as described in `data_raw/README.md`.

```bash
# 1. (optional) download the current official source files
python scripts/ingest/fetch_official_sources.py

# 2. (optional) clean raw workbooks -> tidy CSVs, then build model inputs
python scripts/clean/extract_abs_motor_vehicle_census.py
python scripts/clean/extract_road_vehicles.py
python scripts/clean/extract_energy_table_o.py
python scripts/clean/extract_yearbook.py
python scripts/build/build_model_inputs.py

# 3. modelling pipeline
python scripts/modeling/marginal_benchmark.py
python scripts/modeling/final_marginal_envelopes.py
python scripts/modeling/dependence_model.py
python scripts/modeling/joint_monte_carlo.py
python scripts/modeling/risk_decision_metrics.py
python scripts/modeling/risk_robustness.py
python scripts/modeling/intervention_levers.py

# 4. publication figures (written to figures/)
python scripts/figures/generate_manuscript_figures.py
```

Key fixed parameters: 20,000 Monte Carlo draws (seed 20260620), Student-*t* copula
with 4 degrees of freedom, and square-root-of-horizon scaling of the marginal
forecast residual quantiles.

### Figures used in the manuscript

`scripts/figures/generate_manuscript_figures.py` writes to `figures/` (with the
appendix panels under `figures/supplementary/`). The outputs correspond to the
published figures as follows:

| Script output | Manuscript figure |
|---|---|
| `fig02_marginal_envelope_panel.pdf` | Figure 2 |
| `fig03_dependence_matrix.pdf` | Figure 3 |
| `fig04_central_joint_probabilities.pdf` | Figure 4 |
| `fig05_downside_risk_panel.pdf` | Figure 5 |
| `fig06_robustness_stability.pdf` | Figure 6 |
| `fig07_driver_shift_ranking.pdf` | Figure 7 |
| `fig08_driver_shift_components.pdf` | Figure 8 |
| `supplementary/supp01_observed_context.pdf` | Figure A.1 |
| `supplementary/supp02_model_forecasts_with_final_envelopes.pdf` | Figure A.2 |
| `supplementary/supp03_marginal_benchmark_mape.pdf` | Figure A.3 |
| `supplementary/supp04_central_joint_probability_trajectories.pdf` | Figure A.4 |
| `supplementary/supp05_risk_and_intervention_detail.pdf` | Figure A.5 |

Figure 1 is a conceptual workflow diagram prepared separately. The analysis code
reproduces Figures 2–8 and the supplementary figures listed above.

## Data sources

The processed tables were derived from official public sources. The original
workbooks are not redistributed here because some contain embedded links and document
metadata that are unrelated to the analysis. Official landing pages and download URLs
are recorded in `data_sources/source_registry.csv`.

- **Australian Bureau of Statistics (ABS)** — Motor Vehicle Census.
- **Bureau of Infrastructure and Transport Research Economics (BITRE)** — Road
  Vehicles, Australia; Yearbook (transport energy, emissions, and activity tables).
- **Australian Energy Statistics (AES)** — Table O, electricity generation by fuel
  type (NSW).

ABS and Australian Energy Statistics material is available under the Creative Commons
Attribution 4.0 International licence. BITRE material remains subject to the terms
attached to each publication or government data catalogue record. Provider logos, the
Commonwealth Coat of Arms, and third-party material are not relicensed by this
repository.

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`). Processed
data retain the attribution requirements of their original sources. The MIT Licence
does not replace or extend those terms.

## Citation

If you use this code or data, please cite the manuscript (citation details to be added
on publication).
