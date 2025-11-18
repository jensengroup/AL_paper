# Finding Drug Candidate Hits With a Hundred Samples: Ultra-low Data Screening With Active Learning

This repository contains data and analysis scripts for the paper *[Finding Drug Candidate Hits With a Hundred Samples: Ultra-low Data Screening With Active Learning]([https://doi.org/10.26434/chemrxiv-2025-jlml](https://chemistry-europe.onlinelibrary.wiley.com/share/JKNYRCHHIJFDMHR4GJRG?target=10.1002/ceur.202500134))*.

## Contents

- **`Grapher.ipynb`** – Generates figures and performs analysis for the main paper.
- **`GrapherSIDtp.ipynb`** – Analyzes and visualizes data for the DTP section of the Supporting Information.
- **`GrapherSIDds10.ipynb`** – Analyzes and visualizes data for the DDS10 section of the Supporting Information.
- **`GrapherSIDds10.ipynb`** - Utility functions for generating the graphs. 

## CSV Result Files

All results are stored in `.csv` files. Each file contains the following columns:

- `replicate`: Current replicate number (1–30).
- `rank`: Active learning iteration (1–5).
- `top-100/1320 model`: Number of true top-100/1320 molecules predicted by the model (not used in current figures).
- `top-100/1320 acquired`: Fraction of top-100/1320 molecules found in the acquired set (e.g., `0.003030 * 1320 ≈ 4` molecules for the DTP dataset).

## Directory Overview

- **`all/`** – Results *without* PADRE data augmentation. `10k/` corresponds to DDS10; `130k/` to DTP.  
- **`all_PADRE/`** – Results *with* PADRE data augmentation. `10k/` is DDS10; `130k/` is DTP.  
- **`01_BestCombosDTP/`** – A curated collection of the best descriptor/model combinations for the **DTP** dataset, as shown in the main paper.  
- **`02_BestCombosDDS10/`** – Same as above, but for the **DDS10** dataset.
- **`starting_SMILE_sets/`** - The starting sets for the active learning experiments.

## Reproducing Experiments

This project uses the [MDRMF](https://github.com/MolinDiscovery/MDRMF) package to run active learning experiments from YAML configuration files.

To reproduce an experiment (e.g., acquisition function tests on DTP):

1. Locate the relevant settings file, for example:

    ```
    all_PADRE/130k/acquisition130k_CDDD_pair-240920-162255-5d/settings.yaml
    ```

2. Install the MDRMF package. See [MDRMF](https://github.com/MolinDiscovery/MDRMF) for details.
3. Download dataset(s) (see below)
4. Run the experiment with:

    ```bash
    python -m MDRMF.experimenter path/to/settings.yaml
    ```

💿 **Datasets** can be downloaded from [here](https://sid.erda.dk/sharelink/dVyPBnFi3U).

## Starting Molecule Sets

The 30 Molecule starting sets are available in the setting.yaml files. They are placed at the very top under `unique_initial_sample`. For convenience the SMILES are provided as lists in **starting_SMILE_sets/** along with the starting sets for the data enrichment tests.

These directories contain lists of SMILES including the molecules used for enrichment.

---

> 📦 Built with [MDRMF](https://github.com/MolinDiscovery/MDRMF)
