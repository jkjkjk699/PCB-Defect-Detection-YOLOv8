# -*- coding: utf-8 -*-
"""PCB 缺陷检测 —— 训练脚本

用法示例：
    python scripts/train.py --model yolov8n.pt --epochs 50
    python scripts/train.py --model yolov8s.pt --batch 16 --device 0
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv8 for PCB defect detection')
    p.add_argument('--model', default='yolov8n.pt',
                   help='预训练权重，如 yolov8n.pt / yolov8s.pt（迁移学习起点）')
    p.add_argument('--data', default='data/pcb.yaml', help='数据集配置文件路径')
    p.add_argument('--epochs', type=int, default=50, help='训练轮数')
    p.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    p.add_argument('--batch', type=int, default=16, help='批次大小（受显存限制）')
    p.add_argument('--device', default='0', help='GPU 编号，使用 CPU 则填 cpu')
    p.add_argument('--name', default='pcb', help='本次实验名称')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/detect',
        name=args.name,
    )


if __name__ == '__main__':
    main()
