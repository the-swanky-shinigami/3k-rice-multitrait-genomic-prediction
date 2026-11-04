# 3K Rice Multitrait Genomic Prediction

This repository contains the source code and selected data-processing resources accompanying the study:

**Sparse Multi-Trait Genomic Prediction in the 3,000 Rice Genomes: An Empirical Evaluation of Linear Baselines, Auxiliary Helper Phenotypes, and Neural Multi-Task Strategies**

The project investigates genomic prediction across multiple rice traits using linear and multi-task learning approaches, with an empirical evaluation of predictive performance under different modelling strategies.

## Repository Structure

```text
3k-rice-multitrait-genomic-prediction/
│
├── README.md
├── LICENSE
├── environment.yml
│
├── core_engine.py
├── step_master_pipeline.py
├── pipeline_config.json
│
├── data/
│   └── canonical_phenotypes.csv
│
└── supplementary_code/
    ├── generate_master_figures.py
    └── build_notebook.py
```

## Contents

| File | Description |
|---|---|
| `core_engine.py` | Core computational and modelling implementations used in the study. |
| `step_master_pipeline.py` | Main pipeline for executing the genomic prediction workflow. |
| `pipeline_config.json` | Configuration parameters for the analysis. |
| `data/canonical_phenotypes.csv` | Harmonized phenotype dataset used in the analysis. |
| `supplementary_code/generate_master_figures.py` | Script for generating the study figures from the analysis outputs. |
| `supplementary_code/build_notebook.py` | Utility for generating the analysis notebook. |
| `environment.yml` | Software environment and dependency specifications. |

## Data Availability

The harmonized phenotype data used in this study are provided in the `data/` directory.

The original genotype data were obtained from the publicly available 3,000 Rice Genomes resources. Researchers wishing to reproduce the analysis should obtain the corresponding genotype dataset and ensure that the same dataset version, sample identifiers, marker ordering and preprocessing procedures are used.

The original phenotype sources and their processing are described in the associated publication and supplementary resources.

## Reproducibility

The analysis was implemented in Python. The software environment and required dependencies are specified in `environment.yml`.

To reproduce the analysis:

1. Clone this repository.
2. Obtain the required genotype inputs from the original data source.
3. Set up the software environment using the provided environment specification.
4. Review `pipeline_config.json` and ensure that the input paths and analysis settings are correctly configured.
5. Execute the main pipeline using the instructions and entry point specified in the source code.

The pipeline scripts and configuration file provide the implementation of the computational procedures described in the manuscript.

**Note:** Reproduction requires access to the corresponding genotype inputs and the correct input file structure. The phenotype file provided here represents the harmonized phenotype dataset used by the analysis.

## Citation

If you use this repository in your research, please cite the associated publication:

[The final published article citation and DOI will. be mentioned here when available.]

For software citation, please also cite the archived repository release and its DOI once available through Zenodo.

## License

The source code in this repository is distributed under the MIT License. See `LICENSE` for details.

## Contact

For questions concerning the research methodology or repository, please contact the corresponding author listed in the associated publication.
