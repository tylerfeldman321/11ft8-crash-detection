# 11ft8-crash-detection

Analyzing potential crashes at the [11foot8](https://11foot8.com) bridge.

## Setup
To set up the repository, clone the repo, create an environment, and run `pip install -r requirements.txt` to install the required packages.

## Usage

### Inference on Videos
To get timestamps of potential crashes using image discrepancies across the crash bar, run:

```bash
python3 crash_predictor.py <video_file.mp4>
```

### Generating Data for Processing
To batch generate data for model processing, first add a `crash samples` subdirectory in `data`. Each video should be housed in its own directory with a name is a description of the video (i.e. `2021-03-31_Crane-stuck-c161`). The file extension of the video should be `.copy.mp4`. An example of the proper filestructure is `\data\crash\2021-03-31_Crane-stuck-c161\20210331.121000.11foot82b.copy.mp4`.

Then, run:
```bash
python3 generator.py
```

## Authors
Tyler Feldman, Sarah Habib, Rodrigo de Albuquerque, Shaan Gondalia
