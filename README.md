# BedotProject



## Installation

### Option 1: Creating Environment from `environment.yml`

To create a Conda environment from the provided `environment.yml` file, follow these steps:

1. **Download Miniconda**: Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.

2. **Install Miniconda**: Follow the installation instructions provided on the Miniconda website after downloading the installer.

3. **Set up Conda Environment**:
    ```bash
    conda env create -f environment.yml
    ```
    This command will create a new Conda environment based on the specifications listed in the `environment.yml` file.

4. **Activate the Environment**:
    ```bash
    conda activate bedot_project
    ```
    Replace `bedot_project` with the name of the created environment if it's different.

### Option 2: Installing Dependencies from `requirements.txt`

To install the dependencies directly from the `requirements.txt` file, follow these steps:

1. **Download Miniconda**: Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.

2. **Install Miniconda**: Follow the installation instructions provided on the Miniconda website after downloading the installer.

3. **Set up Conda Environment**:
    ```bash
    conda create --name bedot_project python=<python_version>
    ```
    Replace `<python_version>` with the version of Python you want to use (e.g., `3.10`, `3.11`, etc.).

4. **Activate the Environment**:
    ```bash
    conda activate bedot_project
    ```

5. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the necessary packages listed in the `requirements.txt` file into your current Python environment.
