# Improving Interpretability of Deep Learn-ing in COVID-19 Diagnosis with LIME and SHAP


This project aims to build a convolutional neural network (CNN) for COVID-19 diagnosis using chest X-ray images. The model is developed using the VGG16 architecture with transfer learning, and its predictions are explained using two interpretability methods: LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations).

## Overview
The `Best_Model.py` script contains the implementation of a deep learning model to classify chest X-rays into three categories:
- **COVID-19**
- **Viral Pneumonia**
- **Normal**

The model is built on top of the VGG16 architecture, employing transfer learning to improve classification accuracy. Additionally, this project uses explainable AI (XAI) techniques such as LIME and SHAP to make the model's predictions interpretable to healthcare professionals.

This project is based on the seminar paper "Improving Interpretability of Deep Learning in COVID-19 Diagnosis with LIME and SHAP" submitted to Julius-Maximilians-Universität Würzburg, supervised by Prof. Dr. Frédéric Thiesse. The study investigates the interpretability of a CNN model using LIME and SHAP to enhance trust in AI-assisted diagnostics.

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
```

## Dataset
The dataset used for training and testing is a chest X-ray dataset for COVID-19, viral pneumonia, and normal conditions.

### Note:
The dataset exceeds **25MB** in size and contains over **100 files**, which makes it impractical to upload the entire dataset directly to this repository. Instead, you can download the dataset from Kaggle using the link below:

[COVID-19 Image Dataset on Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

The dataset includes chest X-ray images categorized into three classes:
- **COVID-19**
- **Viral Pneumonia**
- **Normal**

The dataset is organized into `train` and `test` directories, each containing subdirectories for the different classes. The dataset used in this study was released by professors at the University of Montreal and contains a total of 317 images. Please ensure you download and organize the dataset correctly before running the script.

## How to Use
1. **Download the Dataset**:
   - Download the dataset from Kaggle and extract it.
   - Ensure that the dataset is organized into `train` and `test` folders as expected.

2. **Run the Script**:
   - To train the model and generate explanations, run the script using Python:
     ```sh
     python Best_Model.py
     ```

3. **Training and Explanation**:
   - The model is trained using the VGG16 pre-trained model with custom top layers for classifying the chest X-rays.
   - After training, LIME and SHAP are used to provide local and global explanations of the model's predictions, which help visualize the important features contributing to each classification.

## Script Structure
The `Best_Model.py` script includes the following main parts:
1. **Data Preprocessing**:
   - Loading and preprocessing the chest X-ray images, including resizing, normalization, and data augmentation.
2. **Model Implementation**:
   - Defining a CNN model based on the VGG16 architecture, employing transfer learning and adding custom classification layers.
3. **Training the Model**:
   - Compiling and training the model using Adam optimizer and categorical cross-entropy loss.
4. **Model Evaluation**:
   - Evaluating model performance on the test dataset.
5. **Interpretability with LIME and SHAP**:
   - Using LIME to generate heatmaps of regions in chest X-rays that influenced the model's decisions.
   - Using SHAP to visualize feature importance at the pixel level, providing detailed insights into the model's reasoning.

## Results
- The model achieved an accuracy of approximately **91%** on the test dataset.
- The LIME and SHAP explanations highlighted medically relevant features, enhancing the model's transparency and helping healthcare professionals trust the model's predictions.

## Example Output
The script generates several outputs during the model evaluation and explanation stages:
- **Accuracy and Loss Curves**: Training and validation performance over epochs.
- **Confusion Matrix**: Performance of the model on different classes.
- **LIME and SHAP Heatmaps**: Visualizations showing the key areas of X-ray images that influenced the model's decisions.

## Future Work
- **Improve Model Performance**: Use a larger dataset and consider other architectures such as ResNet for further improvement.
- **Enhance Interpretability**: Explore additional interpretability techniques like Grad-CAM to gain more insights into the model's decision-making process.
- **Clinical Evaluation**: Collaborate with medical experts to assess the clinical relevance of the highlighted features and further validate the model's applicability in healthcare settings.

## Contact
If you have any questions or suggestions about this project, please feel free to contact me at:
- **Email**: zhaopanteng@gmail.com

## License
This project is a "Seminararbeit". 
