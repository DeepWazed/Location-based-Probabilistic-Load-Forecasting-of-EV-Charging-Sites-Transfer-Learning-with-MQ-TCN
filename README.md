# Location-based Probabilistic Load Forecasting of EV Charging Sites: Deep Transfer Learning with Multi-Quantile Temporal Convolutional Network (MQ-TCN)

Official implementation of the paper:

**Location-based Probabilistic Load Forecasting of EV Charging Sites: Deep Transfer Learning with Multi-Quantile Temporal Convolutional Network (MQ-TCN)**

📄 **Paper:** https://link.springer.com/chapter/10.1007/978-3-032-03281-2_14

---
## Paper Abstract
Electrification of vehicles is a potential way of reducing fossil fuel usage and thus lessening environmental pollution. Electric Vehicles (EVs) of various types for different transport modes (including air, water, and land) are evolving. Moreover, different EV user groups (commuters, commercial or domestic users, drivers) may use different charging infrastructures (public, private, home, and workplace) at various times. Therefore, usage patterns and energy demand are very stochastic. Characterizing and forecasting the charging demand of these diverse EV usage profiles is essential in preventing power outages. Previously developed data-driven load models are limited to specific use cases and locations. None of these models are simultaneously adaptive enough to transfer knowledge of day-ahead forecasting among EV charging sites of diverse locations, trained with limited data, and cost-effective. This article presents a location-based load forecasting of EV charging sites using a deep Multi-Quantile Temporal Convolutional Network (MQ-TCN) to overcome the limitations of earlier models. We conducted our experiments on data from four charging sites, namely Caltech, JPL, Office-1, and NREL, which have diverse EV user types like students, full-time and part-time employees, random visitors, etc. With a Prediction Interval Coverage Probability (PICP) score of 93.62%, our proposed deep MQ-TCN model exhibited a remarkable 28.93% improvement over the XGBoost model for a day-ahead load forecasting at the JPL charging site. By transferring knowledge with the inductive Transfer Learning (TL) approach, the MQ-TCN model achieved a 96.88% PICP score for the load forecasting task at the NREL site using only two weeks of data.

---
## Citation

If you use this repository in your research, please cite:

bibtex
@article{ali2025location,
  title={Location-Based Probabilistic Load Forecasting of Electric Vehicle Charging Sites: Deep Transfer Learning with Multi-quantile Temporal Convolutional Network},
  author={Ali, Mohammad Wazed and Mustafa, Mohammad Asif Ibna and Shuvo, Md. Aukerul Moin and Sick, Bernhard},
  booktitle={Architecture of Computing Systems},
  pages={205--219},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-3-032-03281-2_14}
}

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
