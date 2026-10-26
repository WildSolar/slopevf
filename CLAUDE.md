# slopevf — notes for Claude sessions

View-factor model for the location-resolved irradiation on multi-row bifacial PV arrays on sloped
ground (Python port of the user's MATLAB model, Solar Energy 2026, plus the extensions of the
follow-up paper). `README.md` is the user documentation, `docs/legacy.md` the corrections relative
to the MATLAB code.

## Working on this repository

- Environment: `.venv` (Python 3.10, editable slopevf and perezsky, extras `all`). Tests:
  `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` (about 5 min).
- Radiance (benchmarks, Radiance tests): `$env:RADIANCE_DIR = "<repo>\.venv\Lib\site-packages\pyradiance"`.
  Ray-tracing caches live in `benchmarks/*/work/` (ignored, several hundred MB; rerunning
  everything takes hours).
- The follow-up manuscript lives in the private repository `General_Irradiation_Model`
  (`Publication_Followup/`, next to this one); its figures are read from `benchmarks/*/results/`
  and `examples/results/` here.
- Sky model: the separate package perezsky (`C:\Users\wimm\Documents\packages\PerezSky`,
  github.com/WildSolar/PerezSky). Change it there, not here.
- Releases: bump the version in `pyproject.toml`, `src/slopevf/__init__.py` and `CITATION.cff`;
  publishing a GitHub release archives it on Zenodo and publishes to PyPI
  (`.github/workflows/publish.yml`, trusted publishing).
- git is not on PATH: use GitHub Desktop's bundled
  `%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe`. Commit on `main` with a
  message file (`commit -F`, UTF-8 without BOM); never push — the user pushes from GitHub Desktop.
