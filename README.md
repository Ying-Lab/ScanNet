# ScanNet
ScanNet: Single-cell annotation informed by transcriptional regulation Network via iterative heterogeneous graph learning
<img width="1067" height="590" alt="image" src="https://github.com/user-attachments/assets/eb4ded6a-4ee3-49b7-a9d3-43bbc41ddd56" />

Prerequisites
-----

- Python 3.9.21
- torch 2.5.1+cu118
- cuda >= 11.8 (our cuda is 12.6)

Installation
-----

```bash
git clone https://github.com/Ying-Lab/ScanNet.git
cd ScanNet/ScanNet
conda create -n ScanNet python=3.9
conda activate ScanNet
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install jupyterlab ipykernel
```

Parameters
-----
dataset: Training dataset

cross_protocol: Wheather training across platforms. Default is False.

cuda: Whether using GPU. Default is True.

type_att_size: Attention parameter dimension

epoch: Training epoches. Default is 60.

lr: Initial learning rate. Default is 0.01


Example
-----
Training on Muraro
```bash
# revise args.dataset
parser.add_argument('--dataset', default='Muraro', type=str)
# run train.ipynb
```
