# -*- coding: utf-8 -*-
"""PCB 缺陷检测 —— 推理脚本

用法示例：
    python scripts/predict.py --weights weights/yolov8n_best.pt --source test.jpg
    python scripts/predict.py --weights weights/yolov8s_best.pt --source ./some_folder --conf 0.3
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Run PCB defect detection inference')
    p.add_argument('--weights', default='weights/yolov8n_best.pt', help='训练好的权重文件')
    p.add_argument('--source', required=True, help='待检测的图片 / 文件夹 / 视频路径')
    p.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    p.add_argument('--device', default='0', help='GPU 编号，使用 CPU 则填 cpu')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        device=args.device,
        save=True,
    )
    print('检测完成，可视化结果保存在 runs/detect/predict/ 目录下')


if __name__ == '__main__':
    main()
