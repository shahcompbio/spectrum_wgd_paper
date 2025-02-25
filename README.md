
# MSK SPECTRUM WGD Paper

This repo provides hosts notebooks and a python library packages for the analyses present in [Ongoing genome doubling promotes evolvability and immune dysregulation in ovarian cancer](https://www.biorxiv.org/content/10.1101/2024.07.11.602772v2)

## Setup and environments

Most notebooks use the environment defined in `environments/pip/scdna_general_requirements.txt`.

### Python package

Spectrum specific python code is provided in the spectrumanalysis python package at `./pysrc`.

Recommended installation:

```
cd pysrc/
pip install -e ./
```

## Data

Notebooks reference environment variable `SPECTRUM_PROJECT_DIR` which should point to the raw data.
