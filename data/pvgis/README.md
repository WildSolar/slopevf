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

Source: PVGIS, European Commission Joint Research Centre (https://joint-research-centre.ec.europa.eu/pvgis_en);
the data are reproduced here for reproducibility and remain subject to the PVGIS terms of use (acknowledge
the source). A later download may give a different TMY, since PVGIS data sets are updated.