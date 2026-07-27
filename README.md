# 基于 YOLOv8 的 PCB 缺陷检测

用 YOLOv8 检测 PCB 板上的制造缺陷。给一张 PCB 图，模型框出缺陷的位置、类别和置信度。这是我的一个个人项目，数据集、训练脚本、对比实验和权重都在仓库里。

## 数据集

数据来自 Roboflow Universe 的公开数据集（CC BY 4.0），共 4400 张图，7 类缺陷：

| 类别 | 含义 |
| --- | --- |
| Short_circuit | 短路 |
| damaged | 元器件损坏 |
| lack_of_part | 缺少零件 |
| miss_welding | 漏焊 / 虚焊 |
| redundant | 多余物 |
| slug | 锡渣 |
| spillover | 焊锡溢出 |

数据集太大没放进仓库，自己从 Roboflow 下 YOLOv8 格式，按 `data/pcb.yaml` 里的路径放好就行。

## 实验环境

| 项目 | 配置 |
| --- | --- |
| 平台 | 阿里云 PAI-DSW |
| GPU | NVIDIA A10（24 GB） |
| 框架 | PyTorch 2.10.0 + CUDA 12.8 |
| 检测库 | Ultralytics 8.4.x |
| Python | 3.12 |

## 快速开始

```bash
pip install -r requirements.txt

# 训练，默认 yolov8n，想换大的加 --model yolov8s.pt
python scripts/train.py --model yolov8n.pt --epochs 50

# 推理
python scripts/predict.py --weights weights/yolov8n_best.pt --source 你的图片.jpg
```

## 在线演示

传一张 PCB 图进去就能框出缺陷，置信度阈值可以调。三种用法：

**Streamlit Cloud（永久链接，打开即用）**

<https://pcb-defect-detection-yolov8-4zjx6ozj6chmptmchgflcc.streamlit.app/>

**Google Colab（临时链接）**

仓库里有 `demo_colab.ipynb`，在 Colab 里全部运行，最后一个单元格会给出一个 `*.gradio.live` 链接，72 小时内有效。

**本地跑（Gradio）**

```bash
pip install -r requirements.txt
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

Streamlit 部署：[share.streamlit.io](https://share.streamlit.io) 用 GitHub 登录 → New app → 选仓库、分支 main、Main file path 填 `streamlit_app.py` → Deploy。

## 实验结果

YOLOv8n 和 YOLOv8s 都训 50 轮，输入 640×640：

| 模型 | 参数量 | 计算量 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | 3.01 M | 8.2 GFLOPs | 0.784 | 0.818 | 0.806 | 0.509 |
| YOLOv8s | 11.13 M | 28.7 GFLOPs | 0.803 | 0.799 | 0.821 | 0.510 |

训练曲线：

| YOLOv8n | YOLOv8s |
| :---: | :---: |
| ![](results/yolov8n/results.png) | ![](results/yolov8s/results.png) |

混淆矩阵：

| YOLOv8n | YOLOv8s |
| :---: | :---: |
| ![](results/yolov8n/confusion_matrix.png) | ![](results/yolov8s/confusion_matrix.png) |

PR 曲线：

| YOLOv8n | YOLOv8s |
| :---: | :---: |
| ![](results/yolov8n/BoxPR_curve.png) | ![](results/yolov8s/BoxPR_curve.png) |

检测效果（YOLOv8n 验证集）：

<p align="center">
  <img src="results/yolov8n/val_batch0_labels.jpg" width="640"/>
</p>

## 一点结论

YOLOv8s 比 YOLOv8n 的 mAP50 只高了约 1.5 个点，参数量和计算量却翻了 3.5 倍左右。换更大的模型并没有明显涨点，说明问题不在模型容量，而在数据——锡渣（slug）、多余物（redundant）这几类样本太少（不到 10 个），模型学不到。下一步应该是补这些稀缺类别的数据，而不是继续堆模型。

## 项目结构

```
PCB-Defect-Detection-YOLOv8/
├── README.md
├── LICENSE                   # MIT
├── requirements.txt
├── app.py                    # Gradio 本地演示
├── streamlit_app.py          # Streamlit Cloud 演示
├── demo_colab.ipynb          # Colab 演示
├── data/
│   └── pcb.yaml              # 数据集配置（路径要改成自己的）
├── scripts/
│   ├── train.py
│   └── predict.py
├── results/
│   ├── yolov8n/
│   └── yolov8s/
└── weights/
    ├── yolov8n_best.pt
    └── yolov8s_best.pt
```

## 参考

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- 数据集: Roboflow Universe（CC BY 4.0）
