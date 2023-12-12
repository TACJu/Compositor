## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn (needed by Mask2Former):

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd Compositor_Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup
```bash
conda create --name compositor python=3.9 -y
conda activate compositor
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
git clone https://github.com/TACJu/Compositor.git
cd Compositor
pip install -r requirements.txt
cd Compositor_Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
```