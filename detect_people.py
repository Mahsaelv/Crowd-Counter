#!/usr/bin/env python3
import argparse
import cv2
from ultralytics import YOLO
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(
        description="Count unique people in an image using YOLOv8"
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    return parser.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot load image {args.image}")
        return

    model = YOLO('yolov8n.pt')
    results = model(img, conf=args.conf, iou=args.iou, classes=[0])

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    scores = results[0].boxes.conf.cpu().numpy()

    out = img.copy()
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        pct = int(s * 100)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{pct}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    count = len(boxes)
    cv2.putText(out, f"Detected {count} people", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Save and show
    cv2.imwrite('out.jpg', out)
    print(f"✅ Detected {count} people")


if __name__ == '__main__':
    main()