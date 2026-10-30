# CGRR: Calibration-guided robust reallocation of flood emergency resources

Code, data and manuscript for:

> Mounika D, Kolluri J. *Calibration-guided robust reallocation of emergency resources under evolving flood risk forecasts: a digital twin framework applied to the 2018 Kerala floods.* Submitted to Natural Hazards (Springer).

The framework turns calibrated (conformal) flood risk forecasts into demand ranges, allocates rescue teams, boats and medical units with a robust optimization model on a road network that degrades with flood depth and landslide blockage, and re-plans only when forecasts or roads change enough. It is evaluated on a hindcast of the August 2018 Kerala floods, with the August 2019 floods as an independent test.

## Repository layout

```
data/
  SOURCES.md        where every dataset comes from (read this first)
  external/         small tables transcribed from official sources (committed)
  raw/              large downloads: rasters, shapefiles (not committed)
  processed/        model-ready inputs (not committed; archived on Zenodo)
notebooks/
  01_data_collection.ipynb   Sentinel-1 flood extent, WorldPop exposure, OSM roads
src/                optimization model, trigger rule, simulator, baselines (in progress)
results/            experiment outputs
manuscript/         LaTeX source (Springer Nature template) and compiled PDF
```

## Reproducing

1. `pip install -r requirements.txt`
2. Run `notebooks/01_data_collection.ipynb` in Google Colab (needs a Google Earth Engine project).
3. Download the 2018 landslide inventory (see `data/SOURCES.md`) into `data/raw/`.
4. Experiments: in progress.

## Licence

Code: MIT (see `LICENSE`). Data in `data/external/` is transcribed from public government and NDRF sources and remains subject to their terms; third-party datasets keep their own licences (e.g. the Kerala landslide inventory is CC BY 4.0).

## Contact

Dannuri Mounika — dmounikaresearch@gmail.com · Corresponding author: Johnson Kolluri — johnson.kolluri@sru.edu.in
