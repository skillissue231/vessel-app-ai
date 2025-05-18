# CAM Vessel Segmentation Project

This repository contains the CAM Vessel Segmentation project, which applies a U-Net model with interchangeable ResNet encoders (ResNet18/34/50) to segment blood vessels in CAM assay images.

## Features
- **UNet with ResNet encoders** via `segmentation_models_pytorch`
- **Mixed Precision (AMP)** training for NVIDIA GPUs
- **Automatic checkpointing** and resume training
- **Configurable** batch size, learning rate, and epochs
- **Streamlit** web interface for interactive training and inference
- **Google Colab support** with a runner script
- **ROI-based postprocessing** for vessel analysis

## Directory Structure

```
cam-vessel-ai/
├── data/
│   ├── cam/
│   │   ├── images/
│   │   └── masks/
│   └── retina/
│       ├── images/
│       └── masks/
├── model/
│   └── unet.py
├── utils/
│   ├── dataset.py
│   └── transforms.py
├── train.py
├── train_colab_run.py
├── streamlit_app.py
├── launch.command
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/cam-vessel-ai.git
cd cam-vessel-ai
conda create -n camenv python=3.10 -y
conda activate camenv
pip install -r requirements.txt
```

## Usage

- **Local training:** `python train.py --config config.yaml`
- **Streamlit UI:** `streamlit run streamlit_app.py`
- **Colab:** `python train_colab_run.py`

## License and Contact

For questions or support, contact ahmet.copur@outlook.com.tr
