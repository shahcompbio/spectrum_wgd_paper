
# MSK SPECTRUM WGD Paper

This repo provides hosts notebooks and a python library packages for the analyses present in [Ongoing genome doubling promotes evolvability and immune dysregulation in ovarian cancer](https://www.biorxiv.org/content/10.1101/2024.07.11.602772v2)

## Data visualization
See https://shahcompbio.github.io/spectrum_wgd_paper/cohort_visualization.html for a visualization of the multimodal data collected for this project, including scWGS copy numbers, MEDICC2 trees, SBMClone results, doubleTime results, and scRNA-seq inferCNV results.

## Setup and environments

Most notebooks use the environment defined in `environments/pip/scdna_general_requirements.txt`.

### Python packages

Spectrum specific python code is provided in the spectrumanalysis python package at `./pysrc`.

Recommended installation:

```
cd pysrc/
pip install -e ./
```

## Data

Processed data files and links to raw data are located on the project Synapse page: https://www.synapse.org/Synapse:syn66366718 

Notebooks reference environment variable `SPECTRUM_PROJECT_DIR` which should point to the raw data.

## Repository structure

The files in this repository are organized into directories as follows:

* `analyses/if/mn_rates`: micronuclei rates by slide
* `annotations`: patient-level annotations
* `config`: color configuration used to generate figures
* `environments`: configuration files for python and conda environments used to process DLP and IF data
* `if`: slide-level IF summary statistics
* `metadata/tables`: tables containing sample summary statistics
* `notebooks`: notebooks used to analyze processed data and generate figures and final results
* `pipelines`: pipelines used to process IF and DLP scDNA data
* `pysrc`: Python package used to process data and generate results
* `results`: DLP and in vitro data summary statistics

