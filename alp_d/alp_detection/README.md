# Automatic-Number-Plate-Recognition-YOLOv8
## KUDO'S
Repo source: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/assets/79400407/1af57131-3ada-470a-b798-95fff00254e6

## Data

The example video used in the code can be downloaded at: https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view.

## Model

A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort). 
ALREADY IN THE FOLDER STRUCTURE:

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* With conda:
conda env create -f alp_d\alp_detection\requirements.yml
* Install the project dependencies 

* Run main.py with the sample video file to generate the test.csv file 

this will also trigger functions from util, and at the end of the script add_missing_data.py and then visualize.py are triggered.

* add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python add_missing_data.py
```

* visualize.py file for passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
```python
python visualize.py
```
