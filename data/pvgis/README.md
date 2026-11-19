# PVGIS typical meteorological year, Davos

`pvgis_tmy_davos_v5_3_free.csv`: hourly typical meteorological year (TMY) for Davos (46.838 N, 9.813 E),
used by the examples (`examples/mismatch.py`, `examples/effect_budget.py`, `examples/diffuse_effect.py`)
and the manuscript. Columns (UTC time): air temperature, relative humidity, GHI, DNI, DHI, longwave
radiation, wind speed and direction, pressure; the irradiation is free-horizon (the local horizon is not
applied).

Retrieved with pvlib from the PVGIS 5.3 API:

```python
import pvlib
pvlib.iotools.get_pvgis_tmy(46.838, 9.813, usehorizon=False, map_variables=True,
                            url="https://re.jrc.ec.europa.eu/api/v5_3/")[0].to_csv("pvgis_tmy_davos_v5_3_free.csv")
```

Provenance (PVGIS API metadata, 2026-11-19): radiation database PVGIS-SARAH3, meteorological database ERA5,
years 2005 - 2023, grid elevation 2472 m; the API returned exactly this data (yearly GHI 1326.1 kWh/m2) on that
date. A later download may differ, since PVGIS data sets are updated.

Source: PVGIS, European Commission Joint Research Centre (https://joint-research-centre.ec.europa.eu/pvgis_en),
with radiation data from CM SAF (SARAH-3) and meteorological data from ERA5 (Copernicus Climate Change Service).
The file is reproduced here for reproducibility with attribution to these sources; the terms of PVGIS and of the
underlying data sets apply (the Commission's legal notice licenses content owned by the EU under CC BY 4.0 unless
otherwise indicated).