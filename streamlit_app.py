# -*- coding: utf-8 -*-
"""PCB 缺陷检测在线演示（Streamlit 版）

本地运行：
    streamlit run streamlit_app.py

部署到 Streamlit Community Cloud（免费）：
    1. 用 GitHub 账号登录 https://share.streamlit.io
    2. New app → 选择本仓库 jkjkjk699/PCB-Defect-Detection-YOLOv8
    3. Main file path 填 streamlit_app.py → Deploy
"""
import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

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


@st.cache_resource
def load_model():
    """加载模型并缓存，避免每次交互都重新加载。"""
    weights = "yolov8n.pt"  # 回退：官方预训练权重
    for candidate in ("weights/yolov8n_best.pt", "yolov8n_best.pt"):
        if os.path.exists(candidate):
            weights = candidate
            break
    return YOLO(weights)


st.set_page_config(page_title="PCB 缺陷检测", page_icon="🔍", layout="centered")
st.title("基于 YOLOv8 的 PCB 缺陷检测")
st.caption("上传一张 PCB 图像，模型将自动框出缺陷位置、类别与数量。")

model = load_model()

uploaded = st.file_uploader("上传 PCB 图像", type=["jpg", "jpeg", "png", "bmp"])
conf = st.slider("置信度阈值", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("检测中..."):
        results = model(image_bgr, conf=conf, verbose=False)
    result = results[0]

    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="检测结果", use_container_width=True)

    counts = {}
    for c in result.boxes.cls.tolist():
        name = result.names[int(c)]
        cn = CLASS_NAMES_CN.get(name, name)
        counts[cn] = counts.get(cn, 0) + 1

    if counts:
        st.success("检测到缺陷：" + "，".join(f"{k} {v} 处" for k, v in counts.items()))
    else:
        st.info("未检测到缺陷（该图像可能为合格品）")
else:
    st.info("请先上传一张 PCB 图像开始检测。")
