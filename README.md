# DiffCL: Diffusion Contrastive Learning for Strip Steel Defect Classification

This repository contains the official implementation of **DiffCL**: a **Diffusion Contrastive Learning** (DiffCL) framework for **strip steel defect classification**. DiffCL combines diffusion processes with contrastive learning to improve feature representation, making it particularly effective in challenging industrial applications like defect detection in steel manufacturing.

## Introduction

**DiffCL** is designed to address the challenging task of identifying and classifying defects on strip steel surfaces, where defects often exhibit subtle, fine-grained, and complex patterns. By integrating diffusion processes with contrastive learning, the model enhances the learning of feature representations, enabling better classification accuracy despite the challenges posed by imbalanced datasets and high intra-class variation.

This repository provides a PyTorch-based implementation that replicates the approach described in the paper **DiffCL: Diffusion Contrastive Learning for Strip Steel Defect Classification**. The experiments demonstrate that **DiffCL** consistently outperforms existing defect classification methods on various industrial datasets, including strip steel and PCB defect datasets, as well as on a general natural image dataset.

## Requirements

To run **DiffCL**, you need to install the following dependencies:

- **Python**: >= 3.7
- **PyTorch**: >= 1.8
- **torchvision**: >= 0.9
- **other dependencies**: Please install them via `pip install -r requirements.txt`

## Datasets

We evaluate **DiffCL** on multiple datasets. Here are the datasets used in the experiments:

1. **In-house Steel**: A dataset containing 20 classes and 30,000 images of strip steel defects. The dataset is particularly challenging due to a highly imbalanced class distribution, varying defect scales, and fine-grained similarities between defects. 
   
2. **NEU-CLS**: A dataset with 6 defect types and 14,400 images of steel surface defects. The controlled background simplifies some aspects of the task, but the small size of the defects makes accurate classification difficult.  
   Dataset Link: [https://gitcode.com/open-source-toolkit/768c6]

3. **PKU-Market-PCB**: This dataset includes 6 classes and 21,664 images of printed circuit boards with cluttered backgrounds. The dataset is challenging due to significant scale variation and occlusions, requiring the model to be robust to these environmental complexities.  
   Dataset Link: [https://robotics.pkusz.edu.cn/resources/dataset/]

4. **CIFAR-100**: A general benchmark dataset with 100 classes and 60,000 natural scene images. This dataset is used to evaluate the model's generalization capabilities across a broad range of objects and environments.  
   Dataset Link: [https://www.cs.toronto.edu/~kriz/cifar.html]


## Results

### Comparison with SOTAs on In-house Steel Dataset

| Method       | Accuracy | Recall  | F1-Score | AUROC  | AUPRC  |
|--------------|----------|---------|----------|--------|--------|
| Tip-Adapter  | 57.91%   | 52.39%  | 46.30%   | 92.88% | 63.80% |
| EMO-1M       | 69.18%   | 54.84%  | 55.70%   | 97.15% | 78.95% |
| EMO-6M       | 73.01%   | 60.94%  | 61.37%   | 97.45% | 82.50% |
| SimCLR       | 75.05%   | 60.86%  | 61.59%   | 95.72% | 75.00% |
| SupCon       | 76.55%   | 64.48%  | 66.51%   | 96.33% | 81.17% |
| HCL          | 78.89%   | 66.00%  | 67.72%   | 97.44% | 83.75% |
| Decoupled    | 75.17%   | 67.36%  | 66.30%   | 96.51% | 79.07% |
| ADNCE        | 70.66%   | 65.92%  | 66.89%   | 97.43% | 79.98% |
| GCA          | 71.57%   | 49.94%  | 49.75%   | 97.42% | 80.53% |
| FairKL       | 74.58%   | 60.67%  | 60.78%   | 97.44% | 83.47% |
| DCL          | 74.58%   | 54.48%  | 55.96%   | 96.71% | 80.87% |
| **DiffCL**   | **80.65%** | **71.20%** | **71.87%** | **97.48%** | **86.42%** |


### Comparison with SOTAs on NEU-CLS Dataset

| Method       | Accuracy | Recall  | F1-Score | AUROC  | AUPRC  |
|--------------|----------|---------|----------|--------|--------|
| SimpleNet    | 96.04%   | 96.04%  | 96.04%   | 99.71% | 98.72% |
| SimCLR       | 96.01%   | 96.01%  | 96.02%   | 99.79% | 99.06% |
| SupCon       | 97.85%   | 97.85%  | 97.85%   | 99.93% | 99.69% |
| HCL          | 94.58%   | 94.58%  | 94.60%   | 99.61% | 98.32% |
| Decoupled    | 96.94%   | 96.94%  | 96.95%   | 99.88% | 99.42% |
| ADNCE        | 92.57%   | 92.57%  | 92.58%   | 99.48% | 97.65% |
| GCA          | 86.88%   | 86.88%  | 86.90%   | 97.77% | 94.14% |
| FairKL       | 97.95%   | 97.95%  | 97.95%   | 99.95% | 99.76% |
| DCL          | 96.39%   | 96.39%  | 96.39%   | 99.88% | 99.45% |
| **DiffCL**   | **99.27%** | **99.27%** | **99.27%** | **99.97%** | **99.87%** |


### Comparison with SOTAs on PKU-Market-PCB Dataset

| Method       | Accuracy | Recall  | F1-Score | AUROC  | AUPRC  |
|--------------|----------|---------|----------|--------|--------|
| SimpleNet    | 97.92%   | 97.93%  | 97.91%   | 99.94% | 99.73% |
| SimCLR       | 92.67%   | 92.60%  | 92.63%   | 99.47% | 97.92% |
| SupCon       | 97.72%   | 97.67%  | 97.73%   | 99.89% | 99.57% |
| HCL          | 93.97%   | 93.99%  | 93.95%   | 99.59% | 98.33% |
| Decoupled    | 95.22%   | 95.21%  | 95.22%   | 99.72% | 98.84% |
| ADNCE        | 94.25%   | 94.28%  | 94.26%   | 99.61% | 98.45% |
| GCA          | 89.14%   | 89.38%  | 89.33%   | 98.51% | 95.16% |
| FairKL       | 97.65%   | 97.59%  | 97.66%   | 99.91% | 99.65% |
| DCL          | 94.31%   | 94.26%  | 94.25%   | 99.62% | 98.43% |
| **DiffCL**   | **98.78%** | **98.77%** | **98.79%** | **99.98%** | **99.91%** |


### Comparison with SOTAs on CIFAR-100 Dataset

| Method       | Accuracy | Recall  | F1-Score | AUROC  | AUPRC  |
|--------------|----------|---------|----------|--------|--------|
| SimCLR       | 70.39%   | 70.39%  | 70.56%   | 99.22% | 78.51% |
| SupCon       | 76.68%   | 76.68%  | 76.79%   | 99.12% | 83.73% |
| HCL          | 67.06%   | 67.06%  | 67.30%   | 98.91% | 74.40% |
| DCL          | 68.86%   | 68.86%  | 69.03%   | 99.12% | 76.79% |
| ADNCE        | 70.45%   | 70.45%  | 70.50%   | 99.24% | 79.14% |
| GCA          | 47.98%   | 47.98%  | 47.88%   | 95.81% | 50.24% |
| FairKL       | 70.10%   | 70.10%  | 69.84%   | 92.87% | 66.30% |
| DCL          | 68.76%   | 68.76%  | 68.93%   | 99.15% | 77.31% |
| **DiffCL**   | **78.31%** | **78.31%** | **78.41%** | **99.39%** | **85.87%** |

## Usage

### Clone the Repository

Clone the repository to your local machine:

```bash
https://github.com/Ethan2186/DiffCL.git
cd DiffCL
````

### Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Pretrain the Diffusion Model

Start the pretraining of the diffusion model:

```bash
python pretrain_diffusion.py --dataset <your-dataset>
```

### Pretrain Contrastive Learning

After the diffusion model is pretrained, start the contrastive learning pretraining:

```bash
python pretrain_contrastive.py --dataset <your-dataset>
```

### Linear Training

Finally, perform linear training on the pretrained features:

```bash
python linear_train.py --dataset <your-dataset>
```

