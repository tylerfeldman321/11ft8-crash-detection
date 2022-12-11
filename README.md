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
To get timestamps of potential crashes, run the following command in your terminal or anaconda. Replace <video_file.mp4> with the path to your video file. You must be in the root directory of this repository. You may have to run `python` or `python3` for all of these python commands depending on your installation.

```bash
python crash_predictor.py <video_file.mp4>
```

You can optionally add an argument to specify the probability decision threshold used for the neural network. This will override the default probability threshold that is set in `constants.py`.
```bash
python crash_predictor.py <video_file.mp4> -p=0.5
```

If the path to the video file has a space in it, make sure to include quotes around the filepath. For example if the video is located at `data/crash videos/video.mp4`, then run `python crash_predictor.py "data/crash videos/video.mp4"`

### Training
⚠️ End users should not need to retrain the model. Using the pre-trained model found in `models/mlp.pkl` is recommended. ⚠️

To batch generate data for model training, first add a `crash_samples` subdirectory in `data`. This subdirectory needs to match with the `CRASH_FOLDER` variable in `constants.py`. Each video should be housed in its own directory with a name is a description of the video (i.e. `2021-03-31_Crane-stuck-c161`). The file extension of the video should be `.mp4`. An example of the proper filestructure is `\data\crash_samples\2021-03-31_Crane-stuck-c161\20210331.121000.11foot82b.mp4`.

Then, run:
```bash
python generator.py
```

Once the data is generated, you can retrain the model by running:
```bash
python train.py
```

### Changing settings
If you would like to change some hyperparameters defined, as well as that paths where data and other files are stored, you can edit the constants in the `constants.py` file. However, none of these are necessary to change and should be done so only if familiar with the code.

## Code Structure
`constants.py` - Contains constants defined for paths and other variables.
`crash_predictor.py` - Main file to run to predict crashses on a video file. Contains a CrashPredictor class that has function to train, test, and evaluate the model.
`dataset.py` - Contains file to load and process the dataset.
`generator.py` - Contains code to process video files to generate the dataset.
`experiments.py` - Contains code to run experiments like cross validation and varying the probability decision threshold.
`train.py` - Contains code to train the model on the entire dataset and save the resulting model.

`audio_processing` - Package for audio processing.
`crash_bar_processing` - Package for detecting the crash bar and running structural similarity.
`sign_detection` - Package for detecting the OVERHEIGHT MUST TURN sign.

## Authors
Tyler Feldman, Sarah Habib, Rodrigo de Albuquerque, Shaan Gondalia
