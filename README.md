# 11ft8-crash-detection

Analyzing potential crashes at the [11foot8](https://11foot8.com) bridge.

## Setup

### 1. Clone this repository
To clone this repository, you need git installed on your computer. If it is not installed, download and follow the steps here: https://git-scm.com/downloads.

Once git is installed on your computer, clone this repository by navigating in your terminal to where you'd like to download this code. Then run the following command:

```bash
git clone https://gitlab.oit.duke.edu/tjf40/11ft8-crash-detection.git
```

### 2. Download Python / Anaconda
If you already have python or anaconda downloaded and installed, then skip this step. You will need to download and install python on your device. An easy way of doing this is downloading [anaconda](https://www.anaconda.com/), which also can handle the creation of virtual environments. You can also [download python here](https://www.python.org/downloads/).

### 3. (Optional but Recommended) Create Virtual Environment
Virtual environments are isolated python environments that help separate dependencies for different projects. It isn't necessary but is a good idea to avoid conflicts with other python projects. There are a bunch of options for creating virtual environments. If you are using anaconda, you can follow the [instructions here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#). You can also create a virtual environment with `venv`, with the [instructions available here](https://docs.python.org/3/library/venv.html). After you create your virtual environment, make sure to activate it. You can do this for `venv` with [instructions here](https://docs.python.org/3/library/venv.html#how-venvs-work), or [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) for anaconda. If you are using anaconda, make sure to specify that you want python installed by adding "python" at the end of the command as so: `conda create -n crash-detector python`.

### 4. Install Dependencies into Your Virtual Environment
After creating and activating your environment, change directories into the root of this cloned repository, wherever you downloaded it. Then run `pip install -r requirements.txt` to install the required packages.

## Usage

### Inference on Videos
To get timestamps of potential crashes, run the following command in your terminal or anaconda. You must be in the root directory of this repository. You may have to run `python` or `python3` for all of these python commands depending on your installation.

```bash
python crash_predictor.py <video_file.mp4>
```

You can optionally add an argument to specify the probability decision threshold used for the neural network. This will override the default probability threshold that is set in `constants.py`.
```bash
python crash_predictor.py <video_file.mp4> -p=0.5
```

### Training

End users should not need to retrain the model. Using the pre-trained model found in `models/mlp.pkl` is recommended.
{: .alert .alert-warning}

To batch generate data for model training, first add a `crash samples` subdirectory in `data`. Each video should be housed in its own directory with a name is a description of the video (i.e. `2021-03-31_Crane-stuck-c161`). The file extension of the video should be `.copy.mp4`. An example of the proper filestructure is `\data\crash samples\2021-03-31_Crane-stuck-c161\20210331.121000.11foot82b.copy.mp4`.

Then, run:
```bash
python generator.py
```

Once the data is generated, you can retrain the model by running:
```bash
python3 train.py
```

## Authors
Tyler Feldman, Sarah Habib, Rodrigo de Albuquerque, Shaan Gondalia
