# https://www.cherryservers.com/blog/install-cuda-ubuntu

sudo apt update
sudo apt upgrade 
sudo apt install ubuntu-drivers-common

sudo apt install gcc
gcc -v

# looks like we can direcly install cuda toolkit and the driver will be installed automatically
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# if the above does not work, refer to this solution (https://askubuntu.com/questions/1280205/problem-while-installing-cuda-toolkit-in-ubuntu-18-04)
# have tried the following:
sudo apt clean
sudo apt update
sudo apt purge nvidia-* 
sudo apt autoremove
sudo apt install -y cuda
# this works for me

# install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
# init conda
~/miniconda3/bin/conda init bash

# if needed, install the following for octopi
conda create --name pt23 python=3.8
conda activate pt23
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install hydra-core wandb omegaconf torch pandas tqdm scikit-learn Flask dash opencv-python flask_httpauth
