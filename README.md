
# ECG Heartbeat Classifier 

This project focuses on developing an advanced AI-powered system for  ECG Datapoints analysis and classification, leveraging the MIT-BIH Arrhythmia Dataset. 

By training Machine Learning (SVM, Random Forest, Decision Trees) and Deep Learning (CNN, LSTM) techniques, LSTM, particularly effective in analyzing sequential ECG data,the system achieves over 95% accuracy in detecting cardiac arrhythmias with a false positive rate below 5%. 

The workflow includes data preprocessing, feature extraction, model training, and deployment through a web interface called
**ECG Heartbeat Classifier**, where users can input ECG data and receive classification results based on our trained models. The system categorizes heartbeats into different classifications.
 
## Index


1. [Project Files and Description](#project-files-and-description)
2. [Getting Started](#getting-started)
3. [Research Paper](#research-paper)
4. [Future Work](#future-work)

## Project Files and Description

### Code Files
`ECG-CLASSIFIER.ipynb`: Jupyter notebook for training models using ML and DL techniques on the MIT-BIH Arrhythmia Dataset.

`predict_ecg-1.py`: Python script for loading trained models and predicting ECG classes.

`api.py`: Backend implementation for the web application to process uploaded ECG data and provide classification results.
### Models
`CNN_model.keras`: Trained CNN model for ECG classification.

`RNN_model_updated.keras`: Trained LSTM-based RNN model optimized for sequential data.

`FNN_model.keras`: Fully connected Neural Network (FNN) model for feature-based classification.

### Website
The project also includes a user-friendly web interface **ECG Heartbeat Classifier** to classify heartbeats in uploaded ECG data. 
Users can:

- Register/Login.
- Upload ECG Data: Upload CSV files for analysis.
- View Results: Receive predictions in terms of classes:
     1. Normal Beats
     2. Premature Beats
     3. Supraventricular Beats
     4. Fusion Beats
     5. Unclassified Beats

### Configuration
`Dockerfile`: Docker setup for deploying the project.

`requirements.txt`: Lists the Python dependencies required for the project.
## Getting Started

### 1. Training the Models  
- **Download** the `ECG-CLASSIFIER.ipynb` notebook and the **MIT-BIH Arrhythmia Dataset**, then run the notebook to train the models.  
- The process includes:  
  1. **Importing the dataset** and performing Exploratory Data Analysis (**EDA**).  
  2. **Preprocessing** ECG signals using a low-pass Butterworth filter.  
  3. **Splitting** the dataset into training, validation, and test sets.  
  4. **Training Machine Learning classifiers**: SVM, Random Forest, Decision Trees, and Ensemble models.  
  5. **Training Deep Learning models**: FNN, CNN, LSTM, and Ensemble models.  
  6. **Saving all trained models** for future predictions.  

### 2. Predicting ECG Classes

To predict heartbeat classes from new ECG data, follow these steps:

1. **Install Requirements**:  

Ensure all dependencies are installed by running:  
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Script**:

Use the `predict_ecg-1.py` script to predict heartbeat classes. This script loads the trained models and processes input data for classification:

```bash
streamlit run predict_ecg-1.py
```

### 3. Web Interface  
- Deploy the website for real-time ECG classification:  
    1. **Set up the Docker container** using the provided `Dockerfile`.  
     2. **Access the web application** to upload ECG data and view classification results.  

## Research Paper

### [Deep Learning for the Heart: ECG Analysis and Arrhythmia Detection](https://github.com/Mini-Project-V-Sem/ECG-classification/blob/samruddhi/Deep%20Learning%20for%20the%20Heart%20ECG%20Analysis%20and%20Arrhythmia%20Detection.pdf)
- **Authors**: Samruddhi, Aarya, Sakshi, Bhumi, Mandar
- **Project**: ECG Classification and Arrhythmia Detection
- **Abstract**: 
   This research focuses on using machine learning and deep learning techniques like SVM, RF, DT, CNN, and LSTM to analyze ECG data. The LSTM model achieved the highest accuracy of 95%, demonstrating its efficacy for arrhythmia detection.
- **Highlights**:
   - Real-time ECG signal processing
   - Multi-model evaluation
   - Application of LSTM for improved accuracy
- **Keywords**: ECG Analysis, Arrhythmia Detection, LSTM, Machine Learning

## Future Work  
- Enhance the ensemble approach for achieving higher accuracy.  
- Incorporate additional arrhythmia types to expand classification capabilities.  
- Optimize the web interface to enable faster and more efficient predictions.  
- Explore edge computing for portable ECG analysis devices, enabling real-time predictions on the go.  
