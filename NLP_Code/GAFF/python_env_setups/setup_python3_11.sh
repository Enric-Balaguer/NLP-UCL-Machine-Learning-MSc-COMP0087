#!/bin/bash -l

echo "Job started"

# Change into temporary directory to run work
cd $TMPDIR

#Load Python3.8 to setup python3.8 environment
echo "Load Python node module Python 3.11"
module load python3/3.11

echo "Python version check:"
python --version
python3 --version

echo "Create new Python3.11 environment ..."
#Create new environment
python -m venv /path/of/your/desire/name_of_env
#Activate new environment
source /path/of/your/desire/name_of_env/bin/activate

#Install dependencies from pre-train repository of your choice. In my case GPT-NeoX. 
#echo "Cloning GPT-NeoX repo ..."
#cd /path/you/want/neox/installed/in
#git clone https://github.com/EleutherAI/gpt-neox.git
#cd "gpt-neox"

#Install dependencies
echo "Installing Python dependencies as requested in GPT-NeoX requirements.txt ..."
cd /home/ucabcfj/never_lose_hope/gpt-neox
pip install --upgrade pip
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt # optional, if logging using WandB
pip install -r requirements/requirements-tensorboard.txt # optional, if logging via tensorboard
pip install -r requirements/requirements-flashattention.txt
python ./megatron/fused_kernels/setup.py install # optional, if using fused kernels
