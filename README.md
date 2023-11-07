# Classification on the CIFA10 dataset using a simple CNN
## Setup
#### Create a virtual environment
```
conda create -n env_cifa10 python=3.8
conda activate env_cifa10
```
#### Clone the repository
```
git clone git@github.com:morrisfl/cifar10-classification.git
```
#### Install Pytorch
#### Install pytorch
Depending on your system and compute requirements, you may need to change the command below. See [pytorch.org](https://pytorch.org/get-started/locally/) for more details.
```
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```
#### Install the repository with all dependencies
```
cd cifar10-classification
python -m pip install .
```
If you want to make changes to the code, you can install the repository in editable mode:
```
python -m pip install -e .
```
## Download the CIFA10 Dataset
When you are in the environment navigate to the project root folder. Inside the project root folder run:
```
