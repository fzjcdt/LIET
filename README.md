This repository contains the official PyTorch implementation for our ICCV 2025 paper: **[Mitigating Catastrophic Overfitting in Fast Adversarial Training via Label Information Elimination](https://openaccess.thecvf.com/content/ICCV2025/papers/Pan_Mitigating_Catastrophic_Overfitting_in_Fast_Adversarial_Training_via_Label_Information_ICCV_2025_paper.pdf)**.


## Setup

**1. Clone the repository**
```bash
git clone https://github.com/fzjcdt/LIET.git
cd LIET
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**Environment**

This code has been tested on the following environment:
- OS: Ubuntu 20.04.3
- GPU: Tesla V100
- CUDA: 11.4
- Python: 3.8.10
- PyTorch: 1.10.1
- Torchvision: 0.11.2

## Usage

### Training

To start training the model from scratch, run the main script:
```bash
./main.sh
```
This script will handle the entire training and testing processes as described in the paper.


### Reproducing Figure 1, Figure 2 and Table 1

To reproduce Figure 1, Figure 2, and Table 1 from our paper, please refer to the scripts and instructions provided in the `./paper_figures` directory.


## Citation

If you find this work useful for your research, please consider citing our paper: 
```
@inproceedings{pan2025mitigating,
  title={Mitigating Catastrophic Overfitting in Fast Adversarial Training via Label Information Elimination},
  author={Pan, Chao and Tang, Ke and Li, Qing and Yao, Xin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2991--3000},
  year={2025}
}
```
