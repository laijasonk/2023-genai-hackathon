"""
# Backend

Codebase for the Capgemini GenAI Hackathon.

## Prerequisites

This repository is meant to be run on a machine with the following
packages installed:

* <=Python-3.10
* <=pip-22.0
* <=git-2.34

The examples in this README are written for GNU/Linux, but should be
valid for any machine with standard GNU Coreutils and BusyBox installed
(e.g. BSD, MacOS, WSL, etc.).

### Google Cloud Access

The API requires proper credentials to access outside of GCP. The
recommended method is through gcloud authentication. The instructions
below are for installing gcloud on Ubuntu and most Debian-based
GNU/Linux distributions.

```
# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates gnupg

# Add new source to package list
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Download the public key for Google Cloud and add to keyring
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update package manager and install gcloud
sudo apt update && sudo apt install -y google-cloud-cli

# Check gloud successfully installed
gcloud version

# Authenticate with Google Credentials (copy link into browser)
gcloud auth application-default login

# Set project id
gcloud config set project gen-hybrid-intelligence-team-1
gcloud auth application-default set-quota-project gen-hybrid-intelligence-team-1
```

## Installation

The repository code should run on any Python3 environment as long as it
contains all the dependencies from the `requirements.txt` file. It is
recommended to manage the environment with venv, pipenv, conda, poetry,
etc. for portibility.

### (OPTIONAL) Create a Virtual Environment

For simplicity, the example instructions here uses the python3-venv
module. More robust solutions can be used, but venv has the advantage of
being easily accessible on most operating systems.

```
cd /path/to/repo/backend

# Create a new venv directory
python3 -m venv ./venv

# Activate the newly created venv profile
. venv/bin/activate
```

### (OPTIONAL) Install Dependencies into venv

Assuming the build-in venv module was used, install all the required
dependencies into the environment. The pip3 tool was used here, but it
can vary depending on the chosen environment.

```
cd /path/to/repo/backend

# Activate the venv profile
. venv/bin/activate

# Install the requirements into the newly created venv profile
pip3 install -r ./requirements.txt
```

## Usage

Packages can be used in Python3 scripts and Jupyter notebooks by
importing the modules. Currently, since SetupTools is not used, the
developer must add the repository src to the Python3 path.

```
# Load the modules
sys.path.append(os.path.abspath("./"))
from src.models import vertexai_basic

# Load the config
with open("data/configs/vertexai_basic.json", "r") as fn:
    config = json.load(fn)

# Instantiate chatbot
chatbot = vertexai_basic.VertexAIBasic(config)

# Example loop to get user input
user_input = ""
while not user_input == "quit":
    response = chatbot.add_user_input(user_input)
    print(f"AI: {response}")
    print()
    user_input = input("Human: ")
```

## Documentation

The `./doc` directory in this repository includes API references as well
as a quickstart and step-by-step guide. These files are automatically
generated from Google-style Python docstrings with the `pdoc` package as
shown below.

```
cd /path/to/repo/backend

# OPTIONAL: Activate venv profile if using venv
. venv/bin/activate

# OPTIONAL: Install pdoc if not already installed
pip3 install pdoc

# Run pdoc on localhost
pdoc -d "google" ./src

# Save HTML documentation
pdoc -d "google" -o ./doc ./src
```
"""

__all__ = [
    "models",
    "models.vertexai_basic",
    "models.vertexai_fashion",
    "utils",
    "vectorstore",
]
