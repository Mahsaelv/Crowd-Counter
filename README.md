# Crowd Counter

This repository provides a simple script to detect and count people in an image using YOLOv8.

## Features
- Counts full bodies (`person` class) with YOLOv8.
- Shows confidence (%) above each bounding box.
- Outputs annotated image and prints total.

## Installation
```
1. git clone https://github.com/Mahsaelv/Crowd-Counter.git
```
```
2. cd CrowdCounter
```
```bash
3. pip install -r requirements.txt
```
##Usage

```pycon
python detect_people.py <IMAGE_PATH> [--conf CONF] [--iou IOU]
```

- <IMAGE_PATH>: Path to the input image.

- --conf: Confidence threshold (default: 0.3).

- --iou: IoU threshold for NMS (default: 0.45).

###Example:
```
python detect_people.py sample.jpg --conf 0.25 --iou 0.5
```
The script will generate ```out.jpg``` with annotated boxes and print the count.