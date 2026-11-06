# Processed inputs of the final run (6-7 Nov 2026)

These are the exact files used for all tables and figures of the paper; their SHA-256 prefixes match `results/run_log.txt`.

| File | Content |
|---|---|
| `kerala_flood_exposure_all.csv` | Sentinel-1 flooded area and WorldPop exposed population per district and scene date (2018, 2019) |
| `kerala_travel_time_tau0.csv` | normal road travel times between district points (OpenStreetMap) |
| `kerala_flooded_roads.csv` | road segments flooded on each scene date |
| `route_times_2018.csv`, `route_times_2019.csv` | realized road and boat travel times per date (flooded and landslide-blocked segments removed) |
| `forecasts_2018.csv`, `forecasts_2019.csv` | 6-hourly district risk-class probabilities of the rainfall-based forecaster, with observed class |
| `calibration.csv` | forecasts and observed classes on the calibration periods (Aug-Sep 2017, Jun-Jul 2018) |
| `forecaster_meta.json` | forecaster features, coefficients, thresholds and data splits |
| `static_lowlying.csv` | static low-lying-area index per district |
| `stocks_2018.csv`, `stocks_2019.csv` | resource stocks per hub |
| `landslides_2019.csv` | Puthumala and Kavalappara landslide points |

Not included because of size or because they are re-created from public sources by `notebooks/01_data_collection.ipynb`: Sentinel-1 flood masks (GeoTIFF), the OpenStreetMap road graph (GraphML/GeoPackage), district boundaries, and the 2018 landslide inventory (Hao et al. 2020, DANS, doi:10.17026/dans-x6c-y7x2, downloaded by notebook 02). To reproduce the tables: copy these files to the notebook's `DATA` folder (or set `data_dir` in `configs/experiment.yaml`) and run `python reproduce.py` with `smoke_test: false`.
