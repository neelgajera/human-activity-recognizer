# human-activity-recognizer
this is skeliton based human activity recognizer 
Using Tiny-YOLO oneclass to detect each person in the frame and use 
AlphaPose(resnet50) to get skeleton-pose and then use
ST-GCN model to predict action from every 30 frames 
of each person tracks.

suported actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

## Prerequisites

- Python > 3.7
- Pytorch > 1.7
- opencv-python > 4.5.1
- tqdm > 4.56
- scipy > 1.6
- matplotlib > 3.3.4

## Pre-Trained Models
[models](https://drive.google.com/file/d/1nTRDi0hU5kLEldrJbZtteynzJWObKwOL/view?usp=sharing)
download this file and extrect to project directory

## Basic Use
* use web cam as a input
 ```
 python det.py -C 0
 ```
 * pyqt based app 
```python app.py```

* input video as a input
```
python det.py 
```
* run in colab without install any dependencies
```Untitled10.ipynb```


## Reference

- AlphaPose : https://github.com/Amanbhandula/AlphaPose
- ST-GCN : https://github.com/yysijie/st-gcn
 
