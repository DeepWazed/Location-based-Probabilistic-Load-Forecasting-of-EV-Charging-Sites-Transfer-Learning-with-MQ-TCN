# Location-based Probabilistic Load Forecasting of EV Charging Sites: Deep Transfer Learning with Multi-Quantile Temporal Convolutional Network (MQ-TCN)

Official implementation of the paper:

**Location-based Probabilistic Load Forecasting of EV Charging Sites: Deep Transfer Learning with Multi-Quantile Temporal Convolutional Network (MQ-TCN)**

📄 **Paper:** https://link.springer.com/chapter/10.1007/978-3-032-03281-2_14

---

## Overview

Accurate forecasting of Electric Vehicle (EV) charging demand is becoming increasingly important for reliable power system operation, charging infrastructure planning, and energy management. The stochastic behavior of EV charging caused by diverse user groups, charging locations, and usage patterns makes this forecasting task particularly challenging.

This repository contains the implementation of a **Multi-Quantile Temporal Convolutional Network (MQ-TCN)** together with **inductive Transfer Learning (TL)** for **day-ahead probabilistic EV charging load forecasting** across geographically diverse charging stations.

The proposed framework enables knowledge transfer from data-rich charging locations to data-scarce locations, significantly reducing the amount of historical data required while maintaining high forecasting accuracy.

---
## Citation

If you use this repository in your research, please cite:

```bibtex
@article{ali2025location,
  title={Location-Based Probabilistic Load Forecasting of Electric Vehicle Charging Sites: Deep Transfer Learning with Multi-quantile Temporal Convolutional Network},
  author={Ali, Mohammad Wazed and Mustafa, Mohammad Asif Ibna and Shuvo, Md. Aukerul Moin and Sick, Bernhard},
  booktitle={Architecture of Computing Systems},
  pages={205--219},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-3-032-03281-2_14}
}

```
## Paper Abstract

Electric vehicle charging demand varies significantly across charging stations due to differences in user behavior, charging infrastructure, and geographical location. Existing forecasting methods are often designed for specific charging sites and require large amounts of historical data, limiting their applicability to newly deployed or data-scarce charging stations. This work proposes a **location-based probabilistic load forecasting framework** using a **Multi-Quantile Temporal Convolutional Network (MQ-TCN)**. Unlike deterministic forecasting methods, the proposed model estimates multiple prediction quantiles, allowing uncertainty to be quantified through prediction intervals. To improve adaptability across charging stations, the framework incorporates **inductive Transfer Learning**, enabling a model trained on one location to be efficiently fine-tuned for another location using only a small amount of target-site data. Experiments were conducted using publicly available datasets from **Caltech**, **JPL**, **Office-1**, and **NREL** charging stations. The proposed MQ-TCN achieved substantial improvements in probabilistic forecasting performance compared with conventional machine learning approaches and demonstrated strong transferability to unseen charging locations using only two weeks of target data.

---

## Repository Contents

This repository provides the complete implementation used in the paper, including:

* Data preprocessing pipeline
* Feature engineering
* Multi-Quantile Temporal Convolutional Network (MQ-TCN)
* DeepAR
* iTransformer
* Transfer Learning framework
* Model training
* Fine-tuning on target charging stations
* Probabilistic forecasting evaluation


---

## Methodology

The proposed framework consists of the following stages:

1. Data preprocessing and cleaning
2. Feature engineering
3. MQ-TCN model training on a source charging station
4. Transfer Learning to a target charging station
5. Multi-quantile probabilistic prediction
6. Performance evaluation using probabilistic forecasting metrics

---

## Datasets

Experiments were performed using charging demand data collected from multiple EV charging stations with different user profiles:

* Caltech
* JPL
* Office-1
* NREL

These datasets represent diverse charging environments, including university campuses, research laboratories, office buildings, and public charging facilities.

---

## Key Features

* Deep Temporal Convolutional Network architecture
* Multi-quantile probabilistic forecasting
* Prediction interval estimation
* Inductive Transfer Learning
* Day-ahead EV charging load forecasting
* Adaptation to data-scarce charging stations
* Reproducible experimental pipeline

---

## Repository Structure

```text
.
├── data/                 # Dataset
├── preprocessing/         # Data preprocessing scripts
├── models/                # MQ-TCN Models Arechitectures.
├── notebooks/             # Analysis notebooks
├── utils/                 # Utility functions
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DeepWazed/Location-based-Probabilistic-Load-Forecasting-of-EV-Charging-Sites-Transfer-Learning-with-MQ-TCN/tree/main
cd <repository>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
---

## License

This repository is released for academic and research purposes. Please refer to the LICENSE file for licensing details.

---

## Contact

For questions regarding the paper or the implementation, please open a GitHub Issue or contact the corresponding author.

---

## Acknowledgement

If you find this repository useful, please consider starring ⭐ the repository and citing the accompanying paper.
