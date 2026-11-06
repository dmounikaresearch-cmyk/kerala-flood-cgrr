# Calibration-guided robust reallocation of emergency resources under evolving flood risk forecasts: a digital twin framework

Code, data and manuscript for the paper by D. Mounika and J. Kolluri (submitted to *Natural Hazards*, Springer).

The contribution is **not** a new flood prediction model. The paper shows that recalibrated flood forecasts, converted into demand ranges with a stated coverage, give more reliable and more stable emergency resource allocation as a flood evolves. The case studies are the August 2018 Kerala floods, with the August 2019 floods as an independent test.

Pipeline: multisource data → flood risk forecasts → **temperature-scaling calibration** (ECE, Brier, NLL) → **conformal sets** → demand ranges (exposure × vulnerability) → **robust MIP** with inundation- and landslide-aware routing → **forecast-triggered re-planning** inside a hindcast digital twin.

## Structure

```
README.md  requirements.txt  reproduce.py  CITATION.cff  LICENSE
configs/experiment.yaml        all parameters (ASSUMED values flagged)
data/
  SOURCES.md                   provenance of every dataset (read first)
  external/                    small tables transcribed from official sources (NDRF, Govt of Kerala)
  raw/  processed/  splits/    large or derived data (not committed; archived on Zenodo)
src/
  common.py                    constants
  flood_forecasting.py         interface + validation for the forecasting layer
  calibration.py               temperature scaling, ECE/Brier/NLL, reliability bins, forecast skill
  uncertainty.py               split conformal prediction sets
  demand_estimation.py         damage fractions, Gamma calibration, demand ranges (Eq. 1-2)
  preprocessing.py             realized road/boat travel times from flooded roads and landslides
  resource_optimizer.py        robust model (P), stochastic B3, greedy B2/B5, hazard-aware routes
  digital_twin.py              event state, methods, trigger rule, re-planning loop
  evaluation.py                hindcast scoring, Wilcoxon/Holm, stress S1-S4, sensitivity, tables
  synthetic.py                 random data for code tests only
  check_prop1.py               numerical check of Proposition 1
notebooks/
  01_data_collection.ipynb     GFM / Sentinel-1 flood extent, WorldPop exposure, OSM roads (Colab)
  02_experiments.ipynb         all experiments, tables 3-7, figures (Colab)
experiments/                   per-experiment notes and configs
results/{tables,figures,metrics}
manuscript/                    LaTeX (Springer Nature template) and PDF
checkpoints/                   model checkpoints of the forecasting layer (not committed)
```

## Data

| Category | Source | Role |
|---|---|---|
| Flood extent | Copernicus GFM (2015–present); own Sentinel-1 processing as cross-check | ground truth, exposure, flooded roads |
| Flood forecasts | GloFAS v4 reforecasts (river discharge) | forecasting-model input |
| Rainfall | NASA GPM IMERG | forecasting-model input |
| Population | WorldPop 100 m | exposure, demand proxy |
| Roads and facilities | OpenStreetMap | accessibility, routing |
| Landslides | Hao et al. 2020 (2018 inventory, CC BY 4.0) | road closures |
| Response records | NDRF situation reports; Govt of Kerala memorandum | resource stocks; relief-camp check |

Observed data and simulated operational scenarios are kept separate. **Realized demand is a modelled proxy, not observed emergency demand**, and Table 6c tests alternative demand assumptions. All splits are chronological; the 2019 event is never used for calibration or tuning.

## Reproduce

```bash
pip install -r requirements.txt          # PuLP must be 2.9.x
python src/check_prop1.py                # Proposition 1 check
python reproduce.py                      # smoke_test: true -> random data (code check only)
# put real inputs in data/processed/, set smoke_test: false in configs/experiment.yaml, run again
```

Results of the final run (6 Nov 2026, 200 runs per method) are committed in `results/` (tables, metrics, figures, `run_log.txt` with input checksums).

Outputs: `results/tables/table3`–`table7` (CSV, plus LaTeX rows for Table 5), `results/metrics/` (reliability bins, Wilcoxon/Holm, trigger log) and `results/run_log.txt` (timestamps, versions, configuration, input checksums).

## Methods compared

B1 static · B2 proportional · B3 two-stage stochastic · B4 fixed-budget robust · B5 learned dispatcher · **FU** forecast-based (uncalibrated) · **FC** calibration-guided (nominal) · **CGRR** proposed · ablations A1 (fixed-width ranges), A2a/A2b (re-plan every period / never), A3 (no hazard-aware routing).

## Licence and contact

Code MIT; third-party data keep their own licences. Dannuri Mounika — dmounikaresearch@gmail.com · corresponding author Johnson Kolluri — johnson.kolluri@sru.edu.in
