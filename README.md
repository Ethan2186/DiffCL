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



| Method      | Accuracy   | Recall     | F1-Score   | AUROC      | AUPRC      |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Tip-Adapter | 0.5791     | 0.5239     | 0.4630     | 0.9288     | 0.6380     |
| EMO-1M      | 0.6918     | 0.5484     | 0.5570     | 0.9715     | 0.7895     |
| EMO-6M      | 0.7301     | 0.6094     | 0.6137     | 0.9745     | 0.8250     |
| SimCLR      | 0.7505     | 0.6086     | 0.6159     | 0.9572     | 0.7500     |
| SupCon      | 0.7655     | 0.6448     | 0.6651     | 0.9633     | 0.8117     |
| HCL         | 0.7889     | 0.6600     | 0.6772     | 0.9744     | 0.8375     |
| Decoupled   | 0.7517     | 0.6736     | 0.6630     | 0.9651     | 0.7907     |
| ADNCE       | 0.7066     | 0.6592     | 0.6689     | 0.9743     | 0.7998     |
| GCA         | 0.7157     | 0.4994     | 0.4975     | 0.9742     | 0.8053     |
| FairKL      | 0.7458     | 0.6067     | 0.6078     | 0.9744     | 0.8347     |
| DCL         | 0.7458     | 0.5448     | 0.5596     | 0.9671     | 0.8087     |
| **DiffCL**  | **0.8065** | **0.7120** | **0.7187** | **0.9748** | **0.8642** |

#### omparison with generative contrastive learning methods on In-house Steel Dataset

| Method          | Backbone | Epoch | Accuracy |
| --------------------- | ------------ | --------- | ---------------------- |
| Wu et al.             | ResNet-18    | 300       | 0.6741                 |
| Adainf                | ResNet-18    | 1000      | 0.6960                 |
| CLSP-SimCLR           | ResNet-18    | 300       | 0.6794                 |
| CLSP-SimCLR           | ResNet-18    | 1000      | 0.7201                 |
| CLSP-MoCoV2           | ResNet-18    | 1000      | 0.7176                 |
| CLSP-SimCLR           | ResNet-50    | 1000      | 0.7370                 |
| ActGen                | ResNet-50    | 300       | 0.7733                 |
| Real guidance         | ResNet-50    | 300       | 0.7683                 |
| Azizi et al.          | ResNet-50    | 300       | 0.7707                 |
| Da-Fusion             | ResNet-50    | 300       | 0.7626                 |
| DiffCL                | &nbsp;&nbsp; ResNet-18| 300         | **0.7228**    |
| DiffCL                | &nbsp;&nbsp; ResNet-18| 1000        | **0.7546**|
| DiffCL                | &nbsp;&nbsp; ResNet-50| 300         | **0.7747**|
| DiffCL                | &nbsp;&nbsp; ResNet-50| 1000        | **0.7831**|




### Comparison with SOTAs on NEU-CLS Dataset

| Method     | Accuracy   | Recall     | F1-Score   | AUROC      | AUPRC      |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| SimpleNet  | 0.9604     | 0.9604     | 0.9604     | 0.9971     | 0.9872     |
| SimCLR     | 0.9601     | 0.9601     | 0.9602     | 0.9979     | 0.9906     |
| SupCon     | 0.9785     | 0.9785     | 0.9785     | 0.9993     | 0.9969     |
| HCL        | 0.9458     | 0.9458     | 0.9460     | 0.9961     | 0.9832     |
| Decoupled  | 0.9694     | 0.9694     | 0.9695     | 0.9988     | 0.9942     |
| ADNCE      | 0.9257     | 0.9257     | 0.9258     | 0.9948     | 0.9765     |
| GCA        | 0.8688     | 0.8688     | 0.8690     | 0.9777     | 0.9414     |
| FairKL     | 0.9795     | 0.9795     | 0.9795     | 0.9995     | 0.9976     |
| DCL        | 0.9639     | 0.9639     | 0.9639     | 0.9988     | 0.9945     |
| **DiffCL** | **0.9927** | **0.9927** | **0.9927** | **0.9997** | **0.9987** |


### Comparison with SOTAs on PKU-Market-PCB Dataset

| Method     | Accuracy   | Recall     | F1-Score   | AUROC      | AUPRC      |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| SimpleNet  | 0.9792     | 0.9793     | 0.9791     | 0.9994     | 0.9973     |
| SimCLR     | 0.9267     | 0.9260     | 0.9263     | 0.9947     | 0.9792     |
| SupCon     | 0.9772     | 0.9767     | 0.9773     | 0.9989     | 0.9957     |
| HCL        | 0.9397     | 0.9399     | 0.9395     | 0.9959     | 0.9833     |
| Decoupled  | 0.9522     | 0.9521     | 0.9522     | 0.9972     | 0.9884     |
| ADNCE      | 0.9425     | 0.9428     | 0.9426     | 0.9961     | 0.9845     |
| GCA        | 0.8914     | 0.8938     | 0.8933     | 0.9851     | 0.9516     |
| FairKL     | 0.9765     | 0.9759     | 0.9766     | 0.9991     | 0.9965     |
| DCL        | 0.9431     | 0.9426     | 0.9425     | 0.9962     | 0.9843     |
| **DiffCL** | **0.9878** | **0.9877** | **0.9879** | **0.9998** | **0.9991** |


### Comparison with SOTAs on CIFAR-100 Dataset

| Method      | Accuracy   | Recall     | F1-Score   | AUROC      | AUPRC      |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Tip-Adapter | 0.5504     | 0.5504     | 0.5427     | 0.9812     | 0.6020     |
| EMO-1M      | 0.6242     | 0.6242     | 0.6219     | 0.9848     | 0.7009     |
| EMO-6M      | 0.6605     | 0.6605     | 0.6588     | 0.9815     | 0.7299     |
| SimCLR      | 0.7039     | 0.7039     | 0.7056     | 0.9922     | 0.7851     |
| SupCon      | 0.7668     | 0.7668     | 0.7679     | 0.9912     | 0.8373     |
| HCL         | 0.6706     | 0.6706     | 0.6730     | 0.9891     | 0.7440     |
| Decoupled   | 0.6886     | 0.6886     | 0.6903     | 0.9912     | 0.7679     |
| ADNCE       | 0.7045     | 0.7045     | 0.7050     | 0.9924     | 0.7914     |
| GCA         | 0.4798     | 0.4798     | 0.4788     | 0.9581     | 0.5024     |
| **DiffCL**  | **0.7831** | **0.7831** | **0.7841** | **0.9939** | **0.8587** |

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




