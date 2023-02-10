# ACGCN
This is the implementation of __ACGCN: Graph Convolutional Networks for Activity Cliff Prediction Between Matched Molecular Pairs__ (Park et al., 2022)

## Abstract
One of the interesting issues in drug-target interaction studies is the activity cliff (AC), which is usually defined as structurally similar compounds with large differences in activity toward a common target. The AC is of great interest in medicinal chemistry as it may provide clues to understanding complex properties of the target proteins, paving the way for practical applications aimed at the discovery of more potent drugs. In this paper, we propose graph convolutional networks for the prediction of AC and designate the proposed models as ACGCNs (Activity Cliff prediction using Graph Convolutional Networks). The results show that ACGCNs outperform several off-the-shelf methods when predicting ACs of three popular target datasets for Thrombin, Mu opioid receptor, and Melanocortin receptor. Finally, we utilize gradient-weighted class activation mapping to visualize activation weights at nodes in the molecular graphs, demonstrating its potential to contribute to the ability to identify important substructures for molecular docking.

## Model Architecture  
  
In this paper, we propose two models named as ACGCN-MMP, ACGCN-Sub.

### 1. ACGCN-mmp

<p align="center"><img src="https://user-images.githubusercontent.com/63924704/152633421-ddfc811c-e8f3-4f6e-8540-5cc8e691acde.png"  width="750" height="350"/>  

ACGCN-mmp uses two compounds as input, and eachcompound passes through three graph convolution layers and is expressed as a single vector through a readout function. Then, after combining the features through the two FC layers, the relationship of the MMP is predicted by the one FC layer and the output layer.

<br/>
  
### 2. ACGCN-sub

<p align="center"><img src="https://user-images.githubusercontent.com/63924704/152633434-140eeec4-f3b8-4d43-b398-21bce782860c.png"  width="750" height="350"/>

ACGCN-sub uses core, substituent1, and substituent 2 as input. These inputs pass through three graph convolution layers and are expressed as vectors through readout functions. Then, after combining the features through the three FC layers, the relationship of the MMP is predicted by the one FC layer and the output layer.

<br/>

## Project Structure
```
ACGCN
├── data
│   ├── melanocortin_receptor_4_mmps.csv
│   ├── mu_opioid_receptor_mmps.csv
│   └── thrombin_mmps.csv
├── model
│   ├── acgcn_mmp.py
│   └── acgcn_sub.py
├── utils
│   ├── data_loader.py
│   ├── GCNPredictor.py
│   ├── model_utils.py
│   ├── train.py
│   └── util
├── args.py
├── main.py
├── README.md
└── requirements.txt
```

<br/>
  
## Data Description

The data used in this paper is given in `data` folder. The file name is {target_name}_mmps and extension is .csv (comma separated value). Please see the table below for specific description.

| Column | Type | Description |
| ------ | ------ | ------ |
|molregno1|Number|Identification number of the first compound in [ChEMBL](https://www.ebi.ac.uk/chembl/) Database (version 28)
|molregno2|Number|Identification number of the second compound|
|SMILES1|String|[SMILES](https://doi.org/10.1021/ci00057a005) representation of the first compound in a MMP|
|SMILES2|String|SMILES representation of the second compound in a MMP|
|substituent1|String|SMILES representation of substituent of the first compound|
|substituent2|String|SMILES representation of substituent of the second compound|
|core|String|SMILES representation of shared core|
|standard_value_1| Float | <img src="https://latex.codecogs.com/svg.image?K_i" title="K_i" /> value of the first compound|
|standard_value_2| Float | <img src="https://latex.codecogs.com/svg.image?K_i" title="K_i" /> value of the second compound|
|pKi_1| Float | <img src="https://latex.codecogs.com/svg.image?pK_i" title="pK_i" /> value of the first compound where <img src="https://latex.codecogs.com/svg.image?pK_i" title="pK_i" /> = <img src="https://latex.codecogs.com/svg.image?-\log_{10}&space;K_i/1e^9" title="-\log_{10} K_i/1e^9" /> |
|pKi_2| Float | <img src="https://latex.codecogs.com/svg.image?pK_i" title="pK_i" /> value of the second compound |
|delta| Float | Absolute value of difference between <img src="https://latex.codecogs.com/svg.image?pK_1" title="pK_1" /> and <img src="https://latex.codecogs.com/svg.image?pK_2" title="pK_2" />
|label| Binary | 1 if the MMP is MMP-cliffs, otherwise 0


<br/>

## Getting Started

### Installation

- Clone the repository:
```
git clone https://github.com/chunkyun/ACGCN.git
```

- Install required libraries.

```
pip install -r requirements.txt
```

- PyTorch installation. Only this configurations has been tested:
  
  - Python 3.6.7, PyTorch 1.8.1, CUDA 10.1 

```
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
```

---
## Usage
You can train and test the ACGCN model with the code. For example,
you can run the following code to train the ACGCN-MMP model with the Mu opioid receptor datasets:
```
python main.py --model 'acgcn-mmp' --target_name 'mu_opioid_receptor'
```
For another example, to train the ACGCN-sub model for Melanocortin receptor 4, you run the following code:
```
python main.py --model 'acgcn-sub' --target_name 'melanocortin_receptor_4'
```
If you want to train with different hyper-parameters, please check the arguments list.

### Arguments
- --model: ['acgcn-mmp', 'acgcn-sub']
- --target_name: ['thrombin', 'mu_opioid_receptor', 'melanocortin_receptor_4']
- --random_seed: random seed for data split
- --batch_size: batch size
- --early_stopping_patience: early stopping patience
- --weight_decay: weight decay
- --dropout : dropout probability 
- --device : cuda

---
  
## Bibtex

```
@article{park2022acgcn,
  title={ACGCN: Graph Convolutional Networks for Activity Cliff Prediction between Matched Molecular Pairs},
  author={Park, Junhui and Sung, Gaeun and Lee, SeungHyun and Kang, SeungHo and Park, ChunKyun},
  journal={Journal of Chemical Information and Modeling},
  year={2022},
  publisher={ACS Publications}
}
```

