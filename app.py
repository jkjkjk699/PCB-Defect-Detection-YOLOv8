# -*- coding: utf-8 -*-
"""PCB 缺陷检测交互式演示（Gradio）

本地运行：
    pip install -r requirements.txt
    python app.py
    浏览器打开 http://127.0.0.1:7860

部署到 Hugging Face Spaces：
    将 app.py、requirements.txt 以及 weights/yolov8n_best.pt 上传到 Space 即可，
    HF 会自动安装依赖并给出一个公开访问链接。
"""
import os

import cv2
import gradio as gr
from ultralytics import YOLO

# 依次尝试多个权重路径，兼容本地目录结构与 Hugging Face Spaces 根目录部署；
# 都找不到时回退到官方预训练权重，保证演示不崩溃
WEIGHTS = "yolov8n.pt"
for _candidate in ("weights/yolov8n_best.pt", "yolov8n_best.pt"):
    if os.path.exists(_candidate):
        WEIGHTS = _candidate
        break

model = YOLO(WEIGHTS)

# 类别中英文映射（与 data/pcb.yaml 保持一致）
CLASS_NAMES_CN = {
    "Short_circuit": "短路",
    "damaged": "元器件损坏",
    "lack_of_part": "缺少零件",
    "miss_welding": "漏焊/虚焊",
    "redundant": "多余物",
    "slug": "锡渣",
    "spillover": "焊锡溢出",
}


def detect(image, conf_threshold):
    """对上传图像执行检测，返回标注图与文字说明。"""
    if image is None:
        return None, "请先上传一张 PCB 图像"

    results = model(image, conf=conf_threshold, verbose=False)
    result = results[0]

    # result.plot() 返回 BGR 标注图，Gradio 需要 RGB
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    # 统计各类缺陷数量
    counts = {}
    for c in result.boxes.cls.tolist():
        name = result.names[int(c)]
        cn = CLASS_NAMES_CN.get(name, name)
        counts[cn] = counts.get(cn, 0) + 1

    if counts:
        summary = "检测到缺陷：" + "，".join(f"{k} {v} 处" for k, v in counts.items())
    else:
        summary = "未检测到缺陷（该图像可能为合格品，或可提高置信度阈值再试）"

    return annotated_rgb, summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="上传 PCB 图像"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="置信度阈值"),
    ],
    outputs=[
        gr.Image(type="numpy", label="检测结果"),
        gr.Textbox(label="检测说明"),
    ],
    title="基于 YOLOv8 的 PCB 缺陷检测",
    description="上传一张 PCB 图像，模型会自动框出缺陷位置并给出类别与数量。"
                "可调节置信度阈值以过滤低置信度检测框。",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
