# RS-OVC: Open-Vocabulary Counting for Remote-Sensing Data
Pytorch Implementation for the paper RS-OVC: Open-Vocabulary Counting for Remote-Sensing Data - an open-vocabulary object-counting model for remote-sensing data, [ICPR 2026](https://icpr2026.org/).

<p align="center">
  <img src="images/teaser.png" width="800"/>
</p>

**Tamir Shor<sup>1,2</sup>, George Leifman<sup>2</sup>, Genady Beryozkin<sup>2</sup>**
<sup>1</sup>Technion – Israel Institute of Technology <sup>2</sup>Google Research

Correspondence to: **[tamir.shor@campus.technion.ac.il](mailto:tamir.shor@campus.technion.ac.il)**


arXiv preprint: [[pdf]](https://arxiv.org/pdf/2604.08704)

---

## Overview

We present **RS-OVC** -- the first open-vocabulary object counting framework for remote-sensing imagery.
We show our model enables counting of **novel object classes** using **textual and/or visual conditioning**, without retraining.

---

## Set-up

### 1. Clone repository

```bash
git clone https://github.com/tamirshor7/RS-OVC.git
cd RS-OVC
```

### 2. Create environment

```bash
conda create --name rsovc python=3.10
conda activate rsovc
```

---

## Data

We curate a designated dataset from a set of common aerial imagery object counting and detection dataset, and adapt them for our task of novel-class counting. 
### Download Original Datasets
To replicate creation of our curated dataset used in the paper, download the following public dataset
* **NWPU-MOC**
  [https://github.com/lyongo/NWPU-MOC](https://github.com/lyongo/NWPU-MOC)

* **FAIR-1M**
  [https://www.kaggle.com/code/ollypowell/fair1m-satellite-dataset-eda](https://www.kaggle.com/code/ollypowell/fair1m-satellite-dataset-eda)

* **DOTA**
  [https://captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)

* **DIOR**
  [https://huggingface.co/datasets/torchgeo/dior](https://huggingface.co/datasets/torchgeo/dior)

* **RSOC (Building class only)**
  [https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method)

---

### Data Preprocessing

#### NWPU-MOC

```bash
python data/NWPU-MOC/coco_conversion.py --nwpu_root <NWPU_ROOT>/annotations/
python data/coco_to_odvg.py --data_dir data/NWPU-MOC/ --input_name nwpu_train_class_split.json
```

---

#### FAIR-1M

```bash
mkdir -p <FAIR1M_ROOT>/shared_images
cp <FAIR1M_ROOT>/Images/Train/* <FAIR1M_ROOT>/shared_images
cp <FAIR1M_ROOT>/Images/Val/* <FAIR1M_ROOT>/shared_images

python data/FAIR-1M/coco_conversion.py --fair1m_root <FAIR1M_ROOT>
python data/coco_to_odvg.py --data_dir data/FAIR-1M/ --input_name fair1m_train_class_split.json
```

---

#### DOTA

```bash
mkdir -p <DOTA_ROOT>/shared_images
cp <DOTA_ROOT>/train/images/images/* <DOTA_ROOT>/shared_images
cp <DOTA_ROOT>/val/images/images/* <DOTA_ROOT>/shared_images

python data/DOTA/coco_conversion.py --dota_root <DOTA_ROOT>
python data/coco_to_odvg.py --data_dir data/DOTA/ --input_name dota_train_class_split.json
```

---

#### DIOR

```bash
mkdir -p <DIOR_ROOT>/shared_images
cp <DIOR_ROOT>/JPEGImages-trainval/* <DIOR_ROOT>/shared_images
cp <DIOR_ROOT>/JPEGImages-test/* <DIOR_ROOT>/shared_images

python data/DIOR/coco_conversion.py --dior_root <DIOR_ROOT>
python data/coco_to_odvg.py --data_dir data/DIOR/ --input_name dior_train_class_split.json
```

---

#### RSOC (Building)

```bash
python data/RSOC-Building/coco_conversion.py --rsocb_root <RSOCB_ROOT>
python data/coco_to_odvg.py --data_dir data/RSOC-Building/ --input_name rsocb_train_class_split.json
```

---

## Pretrained Weights (CountGD Initialization)

```bash
mkdir checkpoints
```

Download:

[https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI](https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI)

Place at:

```bash
checkpoints/checkpoint_fsc147_best.pth
```

---

## Configure Dataset Paths

Update dataset paths inside:

```
config/datasets_*.json
```

Example (Linux):

```bash
# NWPU-MOC
sed -i -E 's|("root":\s*")data/NWPU-MOC|\1<NWPU_ROOT>|g' config/*.json

# FAIR-1M
sed -i -E 's|("root":\s*")data/FAIR-1M|\1<FAIR1M_ROOT>|g' config/*.json

# DIOR
sed -i -E 's|("root":\s*")data/DIOR|\1<DIOR_ROOT>|g' config/*.json

# DOTA
sed -i -E 's|("root":\s*")data/DOTA|\1<DOTA_ROOT>|g' config/*.json

# RSOC
sed -i -E 's|("root":\s*")\.\./data/RSOC_building|\1<RSOCB_ROOT>|g' config/*.json
```

---

## Training

### RS-OVC (main model)

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
main.py \
--config_file config/cfg_fndd.py \
--datasets config/datasets_shared_fndd.json \
--mode fused
```

If you encounter NCCL issues:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun ...
```

Single-GPU training also works with minimal performance degradation.

---

### Baselines

| Mode      | Description      |
| --------- | ---------------- |
| `fused`   | RS-OVC (ours)    |
| `rs_only` | RS encoder only  |
| `rsft`    | RS finetuning    |
| `countgd` | Original CountGD |

to train any of the baselines from the paper, set the mode argument to either of 'rs_only', 'rsft','countgd' (naming matches naming conventions from the paper).
To train in RSFT mode pass --config_file config/cfg_fndd_rsft.py (additionaly to --mode rsft). To train in RS-Only mode pass --config_file config/cfg_fndd_rs_only.py (additionaly to --mode rs_only). 


Examples:

```bash
# RS-FT
python main.py --config_file config/cfg_fndd_rsft.py --mode rsft ...

# RS-Only
python main.py --config_file config/cfg_fndd_rs_only.py --mode rs_only ...
```

---

## Evaluation

```bash
python main.py \
--config_file config/cfg_<dataset>_<mode>.py \
--datasets config/datasets_<dataset>.json \
--mode fused \
--eval \
--resume <CHECKPOINT_PATH>
```

### Example (NWPU-MOC)

```bash
python main.py \
--config_file config/cfg_nwpu_fused.py \
--datasets config/datasets_nwpu.json \
--mode fused \
--eval \
--resume <CHECKPOINT_PATH>
```

---

## Checkpoints

RS-OVC trained checkpoints:

```bash
wget -O checkpoint.pth https://huggingface.co/tamirshor/RSOVC/resolve/main/checkpoint.pth
```
---

## Acknowledgments
This work builds upon [CountGD](https://github.com/niki-amini-naieni/CountGD/).
Our method extends its architecture and training framework, and parts of this codebase reuse and adapt components from the original implementation. 
We thank the authors for publicly-releasing their code and enabling our work to build-upon theirs.

## Citation

```bibtex
@inproceedings{shor2026rsovc,
  title     = {RS-OVC: Open-Vocabulary Counting for Remote-Sensing Data},
  author    = {Shor, Tamir and Leifman, George and Beryozkin, Genady},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
}
```


