# Predicting Futures, Not a Future: Diverse Next Action Anticipation

Code for my BSc thesis in Data Science and Artificial Intelligence at Leiden University (LIACS).
Supervised by Dr. Hazel Doughty and Kaiting Liu.

📄 **Read the full thesis [here](https://theses.liacs.nl/3922).** 

## What it is About:

Action anticipation models watch a video and predict what the person will do next. Usually they give one answer. But people often have several reasonable next steps.

This project modifies the [CLAM](https://github.com/zeyun-zhong/scalable-anticipation) action anticipation model so it predicts **5 different, plausible next actions** instead of one.

## Branches

Each branch is one step of the thesis, and each builds on the one before it.

| Branch | Contents |
|---|---|
| `master` | Original CLAM model (baseline) |
| `foundation` | Diversity metrics and support multiple predictions |
| `hit-loss` | First attempt using a simple set loss |
| `div-reg` | Diversity penalty, and giving each prediction a target |
| `transition-prior` | Targets based on action order and visible objects |
| `qualitative-study` | **Final model** and the human evaluation |

## Installation

```bash
git clone https://github.com/zagoritis/Diverse-Action-Anticipation.git
cd Diverse-Action-Anticipation
conda env create -f environment.yml
conda activate clam

pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install triton==3.3.1
pip install --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.3.0 fla-core==0.4.1 flash-linear-attention==0.4.1
```

## Data

The project uses [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) with precomputed RGB features from [RULSTM](https://github.com/fpv-iplab/rulstm), in the `.npy` format provided by [TeSTra](https://github.com/zhaoyue-zephyrus/TeSTra#ek100). 

Add them using the following layout:

```
<DATA_ROOT>/epickitchens100/features/
├── rgb_kinetics_bninception/
├── target_perframe/
├── verb_perframe/
└── noun_perframe/
```

The annotation files are already included in `annotations/`.

## Usage

**Train:**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/ek100/default.yaml --opts DATA.DATA_ROOT_PATH <DATA_ROOT>
```

**Evaluate a trained model:**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/ek100/default.yaml --opts DATA.DATA_ROOT_PATH <DATA_ROOT> \
    TRAIN.ENABLE False TEST.ENABLE True TEST.CKPT_PATH <run_name> \
    MODEL.DIVERSE_SET.READOUT_DEDUP True
```

Checkpoints are saved in `checkpoints/<run_name>/`. All settings are in `configs/ek100/default.yaml`. 

Logging uses Weights & Biases, so run `wandb login` first, or add `USE_WANDB False` to the command.

## Citation

This work builds on [CLAM](https://github.com/zeyun-zhong/scalable-anticipation) by Zhong et al.
