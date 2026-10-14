# FaceCheck Local

一个在自己电脑上运行的人脸图像检测工具。选择图片、点击检测，查看模型对“AI 生成”的预测概率，也可以批量导出 CSV。

使用已训练的 **EfficientNet-B0 + DINOv2-S/14 等权概率融合模型**。支持 CPU 和 NVIDIA GPU；图片在本机处理，网页默认只允许本机访问。模型不是司法鉴定工具，对陌生生成器和画质变化的表现没有保证。

## 安装

建议 Python 3.12。下载或克隆本仓库后，在项目目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# NVIDIA GPU：本项目验证过的 PyTorch CUDA 12.8 版本
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_weights.py
```

仅使用CPU时，把PyTorch安装命令中的索引换成 `https://download.pytorch.org/whl/cpu`。PyTorch环境可能需要数GB下载和存储空间；模型文件本身约100MiB。

## 打开网页

Windows双击 **启动检测网页.cmd**，浏览器打开 http://127.0.0.1:8766 。选择或拖入1–20张图片，点击“开始检测”；首次加载模型稍慢。单张最大8MB、2000万像素，总体积最大24MB。

关闭页面后服务仍在本机后台运行。双击 **关闭检测网页.cmd** 停止服务并释放内存/显存。不会开机自启，不开放局域网或公网访问。

其它系统可运行：

```bash
python local_web/server.py
# 浏览器打开 http://127.0.0.1:8766；终端按Ctrl+C停止
```

## 命令行

```powershell
.\.venv\Scripts\python.exe predict.py --input "你的图片目录" --output predictions.csv --device cpu
# NVIDIA GPU：将 --device cpu 改为 --device cuda
```

CSV列为 `filename,fake_probability`。概率接近1表示更倾向于AI生成，接近0表示更倾向于真实；默认0.5阈值。不同目录的图片不能重名。

## 模型下载

权重单独发布在 [Releases](https://github.com/xjwxjwxjwwjl/facecheck-local/releases)，不放进Git历史。`download_weights.py` 固定版本并核对SHA256；已有文件校验不符时不会覆盖。可以离线复制已下载模型到 `weights/facecheck-ensemble.pt`。推理不需要额外联网下载DINO或ImageNet权重。

## 项目结构

```text
local_web/           本地网页、HTTP服务和启动器
models/blend.py      两分支概率融合
vendor/dinov2/      固定版本的官方骨干源码和许可
core.py             预处理与权重加载
predict.py          批量推理
download_weights.py 模型下载及SHA256验证
model_manifest.json 发布权重的大小、校验值和下载地址
MODEL_CARD.md       模型来源、用途和局限
```

本仓库仅发布推理工具与权重，不包含训练图片、数据划分、训练断点或私人路径。代码采用MIT许可；模型和第三方组件的适用条款与来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [MODEL_CARD.md](MODEL_CARD.md)。
