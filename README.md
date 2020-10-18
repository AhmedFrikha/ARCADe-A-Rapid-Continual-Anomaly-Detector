## ARCADe: A Rapid Continual Anomaly Detector (ICPR 2020)

This code repository contains the implementation of the ARCADe algorithm from the paper 
"ARCADe: A Rapid Continual Anomaly Detector" accepted at ICPR 2020 (https://arxiv.org/abs/2008.04042).

<b>Bibtex:
```
@article{frikha2020arcade,
  title={ARCADe: A Rapid Continual Anomaly Detector},
  author={Frikha, Ahmed and Krompa{\ss}, Denis and Tresp, Volker},
  journal={arXiv preprint arXiv:2008.04042},
  year={2020}
}
```
<b>

The code provided in this repository works with three different datasets, and can easily be extended to further datasets. 
To download the raw data for Omniglot and miniImageNet, follow instructions in https://github.com/spiglerg/pyMeta. 
We downloaded CIFAR-FS from https://github.com/kjunelee/MetaOptNet. 

The following Figure illustrates some of our results on the Omniglot dataset. 
ARCADe enables learning up to 100 unseen anomaly detection tasks, i.e. using only examples from their respective normal class, with minimal forgetting (~ 4% accuracy).
<div style="text-align:center">
<img align="center" src="./r_omniglot.svg" width=50% height=50% />
</div>
