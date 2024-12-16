# DEQ EquiformerV2

This is a minimal implementation of the DEQ EquiformerV2 model from [DEQuify your force field](https://openreview.net/forum?id=rynb4Vn8rb).
We hope that this can serve as a rough template on how to DEQuify your own force field.

This code is not enough to reprocude the results in the paper.
Especially the speed improvements require engineering changes to the TorchDEQ solver and the OC20 evaluation code, which are a bit tedious and break compatibility.
The code currently only works for OC20, not MD17/MD22.

## Usage

```bash
python run_deq.py
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
mamba create -n deq python=3.10 -y
mamba activate deq
pip install -r requirements.txt
wandb login
```

```bash
# get the Open Catalyst Project (required for Equiformerv2)
# outdated: git clone git@github.com:Open-Catalyst-Project/ocp.git
# the project has since moved to https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core
# we will use the old version:
# https://github.com/FAIR-Chem/fairchem/blob/v0.1.0
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem
git checkout 4ac64520f1a17c14989f6d933f7e8887df1adc07
pip install -e .

# Get the OCP data 
cd ocp
# Structure to Energy and Forces (S2EF) task

python scripts/download_data.py --task s2ef --split "200k" --num-workers 8 --ref-energy 
python scripts/download_data.py --task s2ef --split "val_id" --num-workers 8 --ref-energy 

# "2M": 3.4GB (17GB uncompressed)
# https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md
# python scripts/download_data.py --task s2ef --split "2M" --num-workers 8 --ref-energy 

# More data splits:
# python scripts/download_data.py --task is2re
# python scripts/download_data.py --task s2ef --split test
cd ..
```
a
## Code Walkthrough

All changes to existing code are marked by `# Change@DEQ`. All additional code is in the `deq` folder.

Changes made
- Recurrent / Variational Dropout added to `nets/equiformer_v2/drop.py` and initialized in `nets/equiformer_v2/transformer_block.py`
- DEQ added to `deq/deq_oc20.py`