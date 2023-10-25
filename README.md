# Anomaly Detection with Hybrid Model

## Overview
This project focuses on building an anomaly detection model using a hybrid approach that combines K-Means clustering with Autoencoder neural networks. The model aims to identify anomalies in complex data where traditional methods might fall short.

## Project Structure
- **Data Preprocessing**: Data preprocessing steps include dimensionality reduction with PCA and t-SNE, data cleaning, and feature scaling.

- **Model Building**: The project includes a hybrid model that combines K-Means clustering and Autoencoders for anomaly detection. The K-Means model is used for clustering, and the Autoencoder is used to reconstruct data for anomaly detection.

- **Hyperparameter Tuning**: Hyperparameters for the hybrid model, such as the number of clusters and encoding dimensions, are tuned for optimal performance.

- **Anomaly Detection**: The model detects anomalies by comparing the reconstruction error with a threshold. Anomalies are data points with a higher reconstruction error.

- **Evaluation Metrics**: Evaluation metrics such as Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index are used to assess the model's performance on unlabeled data. Additionally, dynamic thresholding is used to classify anomalies.

- **Thresholding**: The project discusses the process of dynamic threshold selection to classify anomalies based on the reconstruction error.

- **Visual Inspection**: Visualizations are used to gain insights into anomalies. The project demonstrates visualizations such as scatter plots and heatmaps to understand the data's anomalies.

- **Resampling**: Techniques for addressing class imbalance are discussed, including oversampling anomalies to improve model performance.

- **Model Finalization**: The final model is determined based on the balance between precision and recall. The README will be updated with the details of the finalized model.



## Getting Started
- Download the dataset https://data.mendeley.com/datasets/ztyk4h3v6s/2
- Run the provided Jupyter notebooks or scripts.

## Requirements
- Python (version 3.12)
- libraries needed.
- NumPy: A fundamental library for numerical operations in Python.

TensorFlow: An open-source machine learning framework for building and training neural networks, including Autoencoders.

Keras: A high-level neural networks API, commonly used with TensorFlow for building and training deep learning models.

Scikit-learn: A machine learning library that provides tools for data preprocessing, model building, and evaluation.

Matplotlib: A popular data visualization library used for creating charts and graphs.

Pandas: A data manipulation and analysis library, often used for data preprocessing.

SciPy: A library for scientific and technical computing, which includes various statistical and optimization functions.

t-SNE: The t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm, often used for dimensionality reduction.

Various other standard Python libraries for data handling, file I/O, and other general-purpose tasks


