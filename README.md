### Project Organization

------------

    ├── allen_brain_toolbox <- Toolbox to convert raw spike data to RDM
    │
    ├── data
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │
    ├── db_backup           <- Backup file of the database containing the main experimental results
    │
    ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── experiments         <- Results of GW Alignment under different experimental settings
    │   ├── all_pseudo_abe
    │   ├── each_animal_analysis
    │   ├── exp_btwn_pseudo
    │   ├── exp_btwn_trials
    │   ├── firing_rate_comparison
    │   ├── pseudo_abe
    │   ├── pseudo_mitamura
    │   ├── results
    │   └── takeda_abe_paper
    │
    ├── GW_methods          <- Toolbox to do GW alignment
    │
    ├── neuropixeltools     <- Predecessor of `allen_brain_toolbox`. Should be safe to delete, but keeping it just in case.
    │
    ├── notebooks           <- Trial and error when the analysis method is not yet determined
    │
    ├── readme_img          <- Images used in ReadMe.md
    │
    ├── src                 <- Predecessor of `allen_brain_toolbox`.
    │   └── neurep_gwot_mouse /
    │       ├── neurep_gwot_mouse

    │
    ├── test                <- Predecessor of `allen_brain_toolbox`. Miscellaneous code for preprocessing.
    │
    ├── VISp_pseudo_results <- Initial results from alignment with pseudo-mouse in VISp.
    │                       　　Not used in papers or publications.
    │
    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make data` or `make train`
    ├── README.md           <- The top-level README for developers using this project.
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    └── tox.ini             <- tox file with settings for running tox; see tox.readthedocs.io


--------
