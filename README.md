
# Anomaly Detection on Time Series: An Evaluation of Deep Learning Methods.

The goal of this repository is to provide a benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods.


## Implemented Algorithms

| Name               | Paper               | 
|--------------------|---------------------|
| LSTM-AD | [Long short term memory networks for anomaly detection in time series](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56), ESANN 2015  |
| LSTM-ED |[LSTM-based encoder-decoder for multi-sensor anomaly detection](https://arxiv.org/pdf/1607.00148.pdf), ICML 2016|
| Autoencoder | [Outlier detection using replicator neural networks](https://link.springer.com/content/pdf/10.1007%2F3-540-46145-0_17.pdf), DaWaK 2002 |


## Usage

```bash
git clone git://github.com/KDD-OpenSource/DeepADoTS.git  
virtualenv venv -p /usr/bin/python3  
source venv/bin/activate  
pip install -r requirements.txt  
python3 main.py
```
