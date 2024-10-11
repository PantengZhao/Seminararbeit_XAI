# COVID-19 Diagnosis with Explainable AI (XAI)

This project aims to build a convolutional neural network (CNN) for COVID-19 diagnosis using chest X-ray images. The model is developed using the VGG16 architecture with transfer learning, and its predictions are explained using two interpretability methods: LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations).

## Overview
The `Best_Model.py` script contains the implementation of a deep learning model to classify chest X-rays into three categories:
- COVID-19
- Viral Pneumonia
- Normal

The model is built on top of the VGG16 architecture, employing transfer learning to improve classification accuracy. Additionally, this project uses explainable AI (XAI) techniques such as LIME and SHAP to make the model's predictions interpretable to healthcare professionals.

## Requirements
To run the script, you need to install the following Python libraries:

- `tensorflow` (version >= 2.0)
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `shap`
- `lime`
- `opencv-python`

To install the required libraries, you can use the following command:
```sh
pip install tensorflow keras numpy pandas matplotlib shap lime opencv-python
