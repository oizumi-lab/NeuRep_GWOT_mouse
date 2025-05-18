# NeuRep_GWOT Analysis (mouse)

## Pipeline
1. Download Spike-Count Data from [Allen Brain Observatory Visual Coding - Neuropixels dataset](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html)
    - Reference: [`example/01_load_spike-data_using_AllenSDK.ipynb`](example/01_load_spike-data_using_AllenSDK.ipynb)
    - Download spike data using AllenSDK
    - Collect data from 8 brain regions (VISp, VISrl, VISl, VISal, VISpm, VISam, LGd, CA1) across 32 individual mice
    - Record neural activity in response to 3 types of stimuli (natural_scenes, natural_movie_one, natural_movie_three)

2. Main Analysis
    - Reference: [`example/02_execute_unsupervised_alignment.ipynb`](example/02_execute_unsupervised_alignment.ipynb)
    - Create Representational Dissimilarity Matrices (RDM)
    - Compute GWOT between two pseudo-mice/individual mice
    - Visualize results with heatmaps, dendrograms, and swarm plots


### Project Organization

------------

    ├── config /             <- Configuration files (YAML) for preprocessing (creating RDM) pipelines
    │
    ├── example /                                   <- Example notebooks for experiment procedure
    │   ├── 01_load_spike-data_using_AllenSDK.ipynb  <- Download Spike-Count Data using AllenSDK
    │   └── 02_execute_unsupervised_alignment.ipynb  <- Main Analysis
    │
    ├── GW_methods /         <- Toolbox for Gromov-Wasserstein Optimal Transport
    │
    ├── scripts /            <- Main analysis scripts
    │   ├── ta_paper_group_alignment.py    <- Group alignment implementation
    │   ├── ta_paper_group_evaluation.py   <- Evaluation for group alignment
    │   ├── ta_paper_ind_alignment.py      <- Individual alignment implementation
    │   └── ta_paper_ind_evaluation.py     <- Evaluation for individual alignment
    │
    ├── session_split /      <- Session split information for experiments
    │   ├── pairs_dict_ind.json            <- Individual mouse pair definitions
    │   ├── session_split_15.csv           <- Session split for 15-mouse experiments
    │   ├── session_split_4.csv            <- Session split for 4-mouse experiments
    │   └── session_split_8.csv            <- Session split for 8-mouse experiments
    │
    ├── setting_files /      <- Setting file for running multiple experiments simultaneously
    │   ├── dummy_group_alignment_setting.csv
    │   ├── dummy_ind_alignment_setting.csv
    │   ├── ta_paper_group_alignment_setting.csv
    │   └── ta_paper_ind_alignment_setting.csv
    │
    ├── src /                 <- Source code for this project
    │   └── neurep_gwot_mouse /
    │       ├── alignment               <- Alignment algorithm implementations
    │       └── allen_brain_toolbox     <- Tools for processing Allen Brain data
    │
    ├── README.md            <- The top-level README for users of this project
    ├── pyproject.toml       <- Project dependencies and configuration
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment
    └── uv.lock              <- Lock file for dependencies

--------
