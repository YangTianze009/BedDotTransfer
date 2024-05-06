# BedotProject

## Installation

### Installing Dependencies from `requirements.txt`

To install the dependencies directly from the `requirements.txt` file, follow these steps:

1. **Download Miniconda**: Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.

2. **Install Miniconda**: Follow the installation instructions provided on the Miniconda website after downloading the installer.

3. **Set up Conda Environment**:
    ```bash
    conda create --name bedot_project python=<python_version>
    ```
    Replace `<python_version>` with the version of Python you want to use (e.g., `3.10`, `3.11`, etc.) 
    
    **NOTE** : `<python_version>=3.11` is preferred.

4. **Activate the Environment**:
    ```bash
    conda activate bedot_project
    ```

5. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the necessary packages listed in the `requirements.txt` file into your current Python environment.

## Overview of Code
```bash
├── Notebook
├── __pycache__
├── datasets
│   ├── mit-bih
│   │   ├── test
│   │   ├── train
│   │   ├── valid
│   │   └── win
│   ├── real_BSG_data
│   │   └── envelope_data
│   ├── stable_noise00
│   │   └── envelope_data
│   │       ├── 0_04
│   │       └── 0_040
│   ├── stable_noise02
│   │   └── envelope_data
│   ├── stable_noise04
│   │   └── envelope_data
│   ├── stable_noise06
│   │   └── envelope_data
│   ├── stable_noise08
│   │   └── envelope_data
│   ├── stable_noise10
│   │   └── envelope_data
│   ├── unstable_noise00
│   ├── unstable_noise02
│   ├── unstable_noise04
│   ├── unstable_noise06
│   ├── unstable_noise08
│   └── unstable_noise10
├── figures
├── logs
├── outputs
├── results
│   ├── results_stable_BSG(00)_50
│   ├── results_stable_BSG(01)_20
│   ├── results_stable_BSG(02)_0.45_0.3_25
│   ├── results_stable_BSG(03)_25_smooth
│   ├── results_stable_BSG(04)
│   ├── results_stable_BSG(05)
│   ├── results_stable_BSG(06)
│   ├── results_stable_BSG(07)
│   ├── results_stable_BSG(08)
│   ├── results_stable_BSG(09)
│   ├── results_stable_BSG(10)
│   ├── results_stable_BSG(11)
│   ├── results_stable_train
│   └── results_unstable_train
└── src
    ├── __pycache__
    ├── envelope_utils # Contains the code for creating enveloped data
    ├── models # Contains Model (Deep Learning Based & Rule Based)
    └── real_data_utils # Contains the Files came with Real BCG Data
```

``` bash
src
├── __init__.py
├── envelope_utils
│   ├── DSP_func.py
│   ├── __init__.py
│   ├── combine_all.py
│   ├── combine_data.py
│   ├── envelope_extraction.py
│   └── envelope_extraction_modified.py
├── models
│   ├── __init__.py
│   ├── deeplearn.py 
│   └── nonparam.py
├── real_data_utils
│   ├── __init__.py
│   ├── main_plot.py
│   ├── readme.docx
│   ├── readme.pdf
│   ├── utils.py
│   └── view_data.py
├── datautils.py
├── ecg_peak_detect.py
├── evaluators.py
├── peak_classification.py
├── signal_classification.py
└── simu_peak_detect.py
```

1. 