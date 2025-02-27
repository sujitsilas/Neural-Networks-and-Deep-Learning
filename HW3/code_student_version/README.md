# Python Environment Setup Instructions

## Requirements
- Python 3.10 or higher installed on your system (Recommended to use Python 3.10 but not required)
    1. If you are using a different version of python the requirements might not work.
    2. In that case, check installed-packages.txt to see what the individual packages are and install them manually.
- The setup below is tested on Python 3.10 and assumes that you will be using Python 3.10.

## Setup Steps

1. Create a virtual environment
```bash
# Create a new virtual environment
python3.10 -m venv venv # The requirments are tested on python 3.10


# Activate the virtual environment
# On Windows:
dlhw\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

2. Install the required packages, before running this make sure your virtual environment is activated, you should see (venv) at the beginning of your terminal prompt.
```bash
pip install -r requirements.txt
```

3. Install jupyter support and make the venv show up in jupyter notebook (this is mainly for those using jupyter notebook inside VS Code)
```bash
pip install ipykernel
python -m ipykernel install --user --name=dlhw"
```

4. If you still don't see the venv in jupyter notebook try reloading VS Code (Command Palette -> "Developer: Reload Window")

