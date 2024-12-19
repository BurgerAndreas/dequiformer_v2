# DEQ EquiformerV2

This is a minimal implementation of the DEQ EquiformerV2 model from [DEQuify your force field](https://openreview.net/forum?id=rynb4Vn8rb).
We hope that this can serve as a rough template on how to DEQuify your own force field.

This code is not enough to reprocude the results in the paper.
Especially the speed improvements require engineering changes to the TorchDEQ solver and the OC20 evaluation code, which are a bit tedious and break compatibility.
The code currently only works for the OC20 dataset, not MD17/MD22.

## Usage
```bash
python main_oc20.py \
    --num-gpus 1 \
    --num-nodes 1 \
    --mode train \
    --config-yml 'deq/deq_200k.yml' \
    --run-dir 'models/deq/200k' \
    --print-every 10 \
    --amp 
```

```bash
python main_oc20.py \
    --num-gpus 1 \
    --num-nodes 1 \
    --mode train \
    --config-yml 'deq/equiformer_200k.yaml' \
    --run-dir 'models/equiformer_v2/200k' \
    --print-every 10 \
    --amp 
```

## Installation

I recommend the mamba environment manager (better version of conda)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
~/miniforge3/bin/mamba
source ~/.bashrc
```

```bash
mamba create -n deqff python=3.10 -y
mamba activate deqff
pip install --upgrade pip
pip install -r requirements.txt
wandb login
```

```bash
# get the right pytorch version for your cuda/rocm version
pip uninstall torch torchvision torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv -y
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# OCP requires torch-geometric<=2.0.4 ?
pip install torch-geometric==2.1.0 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

```bash
# get the Open Catalyst Project
# outdated: git clone git@github.com:Open-Catalyst-Project/ocp.git
# the project has since moved to https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core
# we will use the old version for compatibility with EquiformerV2:
# https://github.com/FAIR-Chem/fairchem/blob/v0.1.0
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem
git checkout 4ac64520f1a17c14989f6d933f7e8887df1adc07
pip install -e .

# Get the OCP data 
# Structure to Energy and Forces (S2EF) task
# rm -rf $(pwd)/fairchem/data/s2ef
python scripts/download_data.py --task s2ef --split "200k" --num-workers 8 --ref-energy 
python scripts/download_data.py --task s2ef --split "val_id" --num-workers 8 --ref-energy 

# "2M": 3.4GB (17GB uncompressed)
# https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md
# python scripts/download_data.py --task s2ef --split "2M" --num-workers 8 --ref-energy 

# More data splits:
# python scripts/download_data.py --task is2re
# python scripts/download_data.py --task s2ef --split test
cd ..

# check
# ls fairchem/data/s2ef/200k/train

# symlink the OCP data to the datasets folder
mkdir -p datasets/oc20
ln -s $(pwd)/fairchem/data/s2ef datasets/oc20/s2ef
```

## Code Walkthrough

All changes to existing code are marked by `Change@DEQ`. All additional code is in the `deq` folder.

Changes made
- Recurrent / Variational Dropout added to `nets/equiformer_v2/drop.py` and initialized in `nets/equiformer_v2/transformer_block.py`
- DEQ added to `deq/deq_oc20.py`

### Not implemented yet
- Fixed-point reuse: change OC20 eval loop to store and pass fixed-point between batches
- Running relaxations / MD simulations with ASE (Atomistic Simulation Environment)
- MD17/MD22 dataset: get training loop from EquiformerV1 and adjust args in `deq/deq_oc20.py` `DEQ_OC20.forward()`
- speedup solver: simplify `torchdeq.solver.anderson` by removing dicts and `batch_masked_mixing`

## Citation

If you use this code, please cite our paper:
```
@inproceedings{
    anonymous2024dequify,
    title={{DEQ}uify your force field: Towards efficient simulations using deep equilibrium models},
    author={Andreas Burger and Lucas Thiede and Alan Aspuru-Guzik and Nandita Vijaykumar},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=rynb4Vn8rb},
    note={under review}
}
```
