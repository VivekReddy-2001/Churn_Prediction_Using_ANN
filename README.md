# Customer Churn Prediction Using Artificial Neural Network (ANN)

This project is aimed at predicting customer churn using a deep learning model (ANN) built with TensorFlow/Keras and deployed using Streamlit. The model is trained to predict whether a customer will churn based on various input features such as credit score, age, geography, and account balance.

## Project Overview

Customer churn prediction is crucial for businesses to retain valuable customers. This project uses an Artificial Neural Network (ANN) model to predict customer churn based on customer data. The model is deployed using Streamlit to create a user-friendly web interface where users can input customer data and receive predictions.

### Features
- Predicts customer churn based on features such as age, geography, credit score, account balance, etc.
- Deployed as a web application using Streamlit.
- Used `OneHotEncoder` for categorical feature encoding and `StandardScaler` for feature scaling.

## Technologies Used
- **Python**: For model development.
- **TensorFlow/Keras**: Used to build and train the ANN model.
- **Pandas**: For data manipulation and processing.
- **Scikit-learn**: For preprocessing (scaling and encoding).
- **Streamlit**: For deploying the web application.
- **Pickle**: For saving and loading the trained model and encoders.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/VivekReddy-2001/Churn_Prediction_Using_ANN.git
cd customer-churn-prediction-ann
```

### 2. Create a Virtual Environment
```bash
# On Windows
python -m venv newenv
newenv\Scripts\activate

# On Mac/Linux
python3 -m venv newenv
source newenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Load the Dataset
Ensure you have the dataset used for training the model. You can use any customer data CSV file with columns like:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

### 5. Training the Model (Optional)
If you want to retrain the model, run the `train.py` script (replace with your actual training script if different):
```bash
python train.py
```

### 6. Running the Streamlit App
```bash
streamlit run app.py
```

This will launch the Streamlit app where you can input customer data and receive a churn prediction.

## Project Files

- **app.py**: The Streamlit app for customer churn prediction.
- **model.h5**: The saved ANN model.
- **One_Hot_encoder_file.pk1**: Pickled `OneHotEncoder` for encoding categorical data.
- **label_encoder_gender.pk1**: Pickled `LabelEncoder` for encoding gender.
- **Scaler.pk1**: Pickled `StandardScaler` for scaling numerical data.
- **train.py**: Script used for training the ANN model (optional, if retraining is needed).
- **requirements.txt**: List of dependencies required to run the project.

## Example Usage

Once the app is running, enter the customer details (such as geography, age, credit score, etc.), and the app will display the probability of churn. If the probability exceeds 0.5, the customer is likely to churn.


## Issues and Contributions

If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

