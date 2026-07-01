# A Unified Open-Set Framework for Scalable PUF-Based Authentication of Heterogeneous IoT Devices

Official implementation of the paper (to appear at IEEE SMC 2026):

> **A Unified Open-Set Framework for Scalable PUF-Based Authentication of Heterogeneous IoT Devices**
> Xin Wang, Peichun Hua, Chip Hong Chang, Wenye Liu, Yue Zheng

A deep learning framework for open-set authentication of heterogeneous Physically Unclonable Function (PUF) devices. The system uses an OpenGAN-based classifier to learn the feature distribution of enrolled devices and robustly reject out-of-distribution impostor devices — without requiring any real impostor samples during training.

## Overview

Traditional PUF authentication schemes either rely on exhaustive CRP databases or per-device digital-twin models, both of which scale poorly and are vulnerable to ML modeling attacks. This framework addresses these limitations by:

- **Unifying heterogeneous PUF types** (SRAM, DRAM, Arbiter) into a single grayscale image representation
- **Open-set rejection** via an adversarially-trained discriminator that learns to distinguish enrolled device features from generator-synthesized pseudo-outliers
- **Single-pass inference**: no repeated stochastic sampling required (unlike BNN-based approaches)
- **Generic secure protocol**: hybrid encryption (RSA + AES-256-GCM) + Bloom filter replay detection



## Architecture

The system operates in three sequential stages:

**Stage 1 — Closed-Set Classifier** (`train_closed_classifier.py`):
Train a ResNet-18 backbone on enrolled device images using cross-entropy loss. After training, the backbone is frozen and used as a feature extractor.

**Stage 2 — OpenGAN Training** (`main.py`):
Train a Generator–Discriminator pair on the 512-dim features extracted by the frozen ResNet. The Generator synthesizes pseudo-outlier features; the Discriminator learns to separate real enrolled features from synthetic ones. No real outlier data is needed.

**Stage 3 — Evaluation** (`roc_test.py`):
Evaluate open-set performance (FAR, FRR, AUROC, F1) across enrolled and unseen impostor devices.

```
PUF Raw Response
      │
      ▼  (LFSR expansion for Arbiter PUF; direct 2D array for SRAM/DRAM)
  W×H Grayscale Image
      │
      ▼
  ResNet-18  ──► 512-dim feature vector
      │
      ├──► [Stage 1]  Cross-entropy classification (K enrolled devices)
      │
      ├──► [Stage 2]  Generator synthesizes pseudo open-set features
      │              Discriminator: real enrolled vs. synthesized
      │
      └──► [Inference] score > τ → accept (enrolled)
                       score ≤ τ → reject (impostor)
```



## Key Results


| PUF Type                    | # Enrolled | Closed-Set Acc | FAR   | FRR   |
| --------------------------- | ---------- | -------------- | ----- | ----- |
| Arbiter PUF (ours)          | 25         | 100%           | 0.35% | 0%    |
| SRAM PUF                    | 40         | 100%           | 0.13% | 0.62% |
| DRAM PUF (noisy)            | 3          | 100%           | 0.33% | 2.04% |
| Heterogeneous (APUF + SRAM) | 45         | 100%           | 0.44% | 0.43% |


End-to-end authentication time on Raspberry Pi 5: **~0.67 s**.

## Project Structure

```
unified_puf/
├── main.py                                        # Stage 2 & 3: OpenGAN training, validation, testing
├── train_closed_classifier.py                     # Stage 1: ResNet-18 closed-set classifier
├── roc_test.py                                    # ROC/AUC/F1 evaluation and metric plots
├── visualization.py                               # Visualization device feature clusters by UMAP
├── model_relevant.py                              # Model definitions (closed-set model, Generator, Discriminator)
├── dataset_relevant.py                            # PUFdataset: custom VisionDataset for PUF images
├── bcnndram.py                                    # BCNN for DRAM dataset
├── bash_for_automated experiments.sh              # Batch experiment runner
├── requirements.txt                               # Python dependencies
├── split_dataset_script/
│   ├── resample_image_size.py                     # Step 1: resize/resample PUF images to target resolution (H×W)
│   ├── spilit_enrolled_outlier_categories.py      # Step 2: partition devices into data_in (enrolled) vs. val_out/test_out (outlier)
│   └── split_within_enrolled.py                   # Step 3: split enrolled devices into train/val/test under "closed/"
└── onnx_conversion/
    ├── onnx_d.py                                  # Export Discriminator to ONNX (for edge deployment)
    └── onnx_res.py                                # Export ResNet classifier to ONNX
```



## Environment Setup

```bash
conda create -n unipuf python=3.11.14
conda activate unipuf
pip install -r requirements.txt
```



## Datasets

All datasets must be downloaded manually from their original sources and placed under `./split_dataset/`.


| Dataset         | Description                                                                                              | Source                                                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SRAM PUF**    | 144 devices, 101 measurements/device                                                                     | [[https://www.ce.cit.tum.de/en/eisec/mitarbeiter/florian-wilde/publication-downloads/]]  |
| **DRAM PUF**    | 5 devices, noisy (20–50°C, 1.27–1.5 V)                                                                   | [[https://orda.shef.ac.uk/articles/dataset/DRAM_Latency-PUF_Responses_and_corresponding_PUF_Phenotype_conversions_Temperature_and_Voltage_environmentally_tested/26977528?file=49383244]]                                                                                 |
| **Arbiter PUF** | Collected on FPGA boards (Nexys4-DDR, ALINX AXU2CGB-E, ALINX AX7Z020B); 32-stage APUF, 100 images/device | (collected in-house)                                                                                                                                                          |




### Expected Directory Structure

After downloading and preprocessing, organize the dataset as follows:

```
split_dataset/
└── size{SIZE}device{NUM_DEVICES}/
    ├── closed/
    │   ├── train/
    │   │   ├── device_0/   # one folder per enrolled device
    │   │   └── device_1/
    │   ├── val/
    │   └── test/
    ├── val_out/            # open-set impostor devices for validation
    └── test_out/           # open-set impostor devices for testing
```

Dataset split ratios: train / val / test = **3:1:1** per enrolled device.

### Dataset Preprocessing

```bash
# 1. Resize images to target resolution
python split_dataset_script/resample_image_size.py

# 2. Partition device categories into enrolled vs. outlier sets
python split_dataset_script/spilit_enrolled_outlier_categories.py

# 3. Split enrolled device images into train/val/test
python split_dataset_script/split_within_enrolled.py
```



## Usage



### Stage 1: Train Closed-Set Classifier

```bash
python train_closed_classifier.py \
    --size 220 \
    --num_c 50 \
    --closed_model res18 \
    --num_epochs 10 \
    --lr 0.0001 \
    --b 128 \
    --device cuda:0
```


| Argument         | Default  | Description                                         |
| ---------------- | -------- | --------------------------------------------------- |
| `--size`         | 50       | Image size identifier (e.g. 220 for 220×200 images) |
| `--num_c`        | 45       | Number of enrolled device classes                   |
| `--closed_model` | `res18`  | Backbone: `res18` or `res34`                        |
| `--num_epochs`   | 10       | Training epochs                                     |
| `--lr`           | 0.0001   | Learning rate (AdamW)                               |
| `--b`            | 256      | Batch size                                          |
| `--device`       | `cuda:0` | CUDA device                                         |


Best model saved to:
`experiments_results/closed{NUM_C}/resnet_weights/best_res18_{NUM_C}puf_size{SIZE}.pth`

---



### Stage 2 & 3: Train OpenGAN + Validate + Test

```bash
python main.py \
    --size 220 \
    --num_d 50 \
    --exp 1 \
    --lr 0.0005 \
    --ngf 256 \
    --b 256 \
    --end_epochs 50 \
    --threshold 0.5 \
    --device cuda:0 \
    --run_train --run_val --run_test
```


| Argument       | Default  | Description                                              |
| -------------- | -------- | -------------------------------------------------------- |
| `--exp`        | 0        | Experiment No.                                           |
| `--size`       | 50       | Dataset image size                                       |
| `--num_d`      | 50       | Number of enrolled classes                               |
| `--lr`         | 0.0005   | GAN learning rate                                        |
| `--ngf`        | 64       | Generator/Discriminator hidden dimension ($n_gf = n_df$) |
| `--b`          | 256      | Batch size                                               |
| `--beta1`      | 0.9      | Adam β₁                                                  |
| `--end_epochs` | 60       | Total training epochs                                    |
| `--threshold`  | 0.5      | Discriminator decision threshold τ                       |
| `--seed`       | 99       | Random seed                                              |
| `--lamba0`     | 1.0      | Loss weight for real (enrolled) samples                  |
| `--lamba1`     | 1.0      | Loss weight for fake (generated) samples                 |
| `--device`     | `cuda:3` | CUDA device                                              |
| `--run_train`  | —        | Run training phase                                       |
| `--run_val`    | —        | Run validation phase                                     |
| `--run_test`   | —        | Run test phase                                           |


Results saved to:
`experiments_results/closed{NUM_D}/exp{EXP}results_size{SIZE}_numd{NUM_D}_seed{SEED}_lr{LR}_ngf{NGF}_beta1{BETA1}_b{BATCH}/`

---



### ROC / AUC Evaluation

```bash
python roc_test.py \
    --size 220 \
    --num_d 50 \
    --exp 1 \
    --device cuda:0
```

Generates ROC curves, discriminator output distribution plots, and saves FAR, FRR, AUROC, F1, Precision, Recall metrics.

## Training Details


| Component           | Configuration                                                    |
| ------------------- | ---------------------------------------------------------------- |
| Closed-set backbone | ResNet-18, 10 epochs, AdamW (lr=1e-4, wd=1e-3)                   |
| GAN training        | 50 epochs, batch size 256, AdamW (β₁=0.9, β₂=0.999)              |
| Generator lr        | 5×10⁻⁴; Discriminator lr: 2×10⁻⁴                                 |
| Hidden dimension    | $n_g = n_d = 256$                                                |
| Loss                | Binary cross-entropy with label smoothing (real=0.95, fake=0.05) |
| Input               | 50×50 grayscale image, replicated to 3 channels                  |




## Deployment (ONNX)

For edge deployment (e.g., Raspberry Pi), export the trained models to ONNX:

```bash
python onnx_conversion/onnx_res.py   # export ResNet classifier
python onnx_conversion/onnx_d.py     # export Discriminator
```


## Citation

If you find this work useful for your research, please consider citing our paper.

### BibTeX

```bibtex
@inproceedings{wang2026unified,
  title={A Unified Open-Set Framework for Scalable PUF-Based Authentication of Heterogeneous IoT Devices},
  author={Wang, Xin and Hua, Peichun and Chang, Chip Hong and Liu, Wenye and Zheng, Yue},
  booktitle={Proc. IEEE Int. Conf. Syst., Man, and Cybern.},
  address={Bellevue, WA, USA},
  month={oct},
  year={2026},
  note={Accepted, to appear}
}
```

### IEEE Word Reference

X. Wang, P. Hua, C. H. Chang, W. Liu, and Y. Zheng, "A Unified Open-Set Framework for Scalable PUF-Based Authentication of Heterogeneous IoT Devices," in *Proc. 2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC)*, Bellevue, WA, USA, Oct. 4-7, 2026, to appear.


## Acknowledgements

This work builds upon [OpenGAN](https://github.com/aimerykong/OpenGAN) (Shu Kong & Deva Ramanan, IEEE PAMI 2022).
We thank the authors for releasing their code.
