# Improving Fair Training under Correlation Shifts

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023
----------------------------------------------------------------------

This repo contains codes used in the ICML 2023 paper: [Improving Fair Training under Correlation Shifts](https://proceedings.mlr.press/v202/roh23a.html)

*Abstract: Model fairness is an essential element for Trustworthy AI. While many techniques for model fairness have been proposed, most of them assume that the training and deployment data distributions are identical, which is often not true in practice. In particular, when the bias between labels and sensitive groups changes, the fairness of the trained model is directly influenced and can worsen. We make two contributions for solving this problem. First, we analytically show that existing in-processing fair algorithms have fundamental limits in accuracy and group fairness. We utilize the notion of correlation shifts between labels and groups, which can explicitly capture the change of the above bias. Second, we propose a novel pre-processing step that samples the input data to reduce correlation shifts and thus enables the in-processing approaches to overcome their limitations. We formulate an optimization problem for adjusting the data ratio among labels and sensitive groups to reflect the shifted correlation. A key benefit of our approach lies in decoupling the roles of pre- and in-processing approaches: correlation adjustment via pre-processing and unfairness mitigation on the processed data via in-processing. Experiments show that our framework effectively improves existing in-processing fair algorithms w.r.t. accuracy and fairness, both on synthetic and real datasets.*


## Setting
This directory is for simulating the proposed pre-processing approach on the 
synthetic dataset. The program needs PyTorch, CVXPY, Jupyter Notebook, and CUDA.

The directory contains a total of 5 files and 1 child directory: 
1 README, 3 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.
The synthetic data contains training set and test set.

## Simulation
To simulate FairBatch, please use the **jupyter notebook** in the directory.

The jupyter notebook will load the data and train the models.
We consider two scenarios: supporting (1) a single metric (DP) and (2) multiple metrics (DP & EO).

Each training shows either in-processing-only baseline or our framework 
(i.e., pre- + in-processing). Note that we use [FairBatch](https://arxiv.org/abs/2012.01696) [Roh et al., ICLR 2021] 
as the in-processing baseline that adaptively adjusts batch ratios for fairness.
When using our pre-processing, we utilize a SDP solver to find the new data ratio. 
The solver is defined in our program.
Experiments are repeated 5 times each.
After the training, the test accuracy and fairness will be shown.

## Other details
The two python files are models.py, utils.py, and FairBatchSampler_Multiple.py.
The models.py contains a logistic regression architecture.
The utils.py contains a total of three functions for 
finding example weight, sampling data, and testing the model performances.
The FairBatchSampler_Multiple.py contains two classes: CustomDataset and FairBatch. 
CustomDataset class defines the dataset, and FairBatch class implements the state-of-the-art 
in-processing technique [Roh et al., ICLR 2021] that adjusts batch ratios for fairness.

The detailed explanations about each component have been written in the codes as comments.
Thanks!

## Reference
```
@inproceedings{pmlr-v202-roh23a,
  title={Improving Fair Training under Correlation Shifts},
  author={Yuji Roh and Kangwook Lee and Steven Euijong Whang and Changho Suh},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={29179--29209},
  year={2023},
  volume={202},
  url={https://proceedings.mlr.press/v202/roh23a.html},
}
```

