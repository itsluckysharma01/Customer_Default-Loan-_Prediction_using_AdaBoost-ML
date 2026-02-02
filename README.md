# Customer Default Prediction using AdaBoost (LOAN)

## 📋 Project Overview

Customer Default Prediction is a critical application used by banks and loan lenders to assess the creditworthiness of potential borrowers. This project implements a machine learning model using **AdaBoost (Adaptive Boosting)**, an ensemble learning technique that combines multiple weak classifiers to create a robust and accurate strong classifier.

The model analyzes various customer attributes to predict whether a borrower is likely to default on their loan, helping financial institutions make informed lending decisions and minimize credit risk.

## 🎯 Project Objectives

- Build a predictive model to identify customers who may default on loans
- Utilize AdaBoost ensemble learning for improved accuracy
- Handle real-world data challenges including missing values and categorical variables
- Achieve high prediction accuracy to support data-driven lending decisions
- Provide clear visualization of model performance

## 🔍 How AdaBoost Works

AdaBoost (Adaptive Boosting) is an ensemble learning method that:

- Iteratively trains a sequence of classifiers
- Each classifier focuses on correcting errors made by the previous one
- Assigns higher weights to misclassified instances
- Combines predictions from all weak learners into a strong final prediction

## 📊 Dataset

The project uses a loan dataset containing customer information and loan characteristics:

- **Source**: [Loans Dataset](https://raw.githubusercontent.com/itsluckysharma01/Datasets/refs/heads/main/gfg_LoanDataset---LoansDatasest.csv)

### Features Include:

- Customer income
- Loan amount
- Home ownership status
- Loan intent
- Loan grade
- Historical default records
- Current loan status (target variable)

## 🛠️ Technologies & Libraries

### Core Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Flask**: Web application framework for the frontend
- **flask-cors**: Cross-Origin Resource Sharing support

### Preprocessing Tools

- `SimpleImputer`: Handling missing values
- `train_test_split`: Splitting data into training and testing sets
- `LabelEncoder`: Encoding categorical variables

### Model

- `AdaBoostClassifier`: Ensemble learning classifier

### Evaluation Metrics

- `accuracy_score`: Model accuracy measurement
- `confusion_matrix`: Detailed performance analysis
- `classification_report`: Comprehensive metrics

## 🚀 Installation & Setup

### Prerequisites

```bash
Python 3.7+
```

### Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 💻 Implementation Steps

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

### 2. Data Loading

Load the dataset from the provided URL or local file.

### 3. Data Preprocessing

- **Handle Missing Values**:
  - Numerical columns filled with mean values
  - Categorical columns filled with mode values
- **Clean Data**: Remove commas and convert string numbers to numeric format
- **Encode Categorical Variables**: Use LabelEncoder for categorical features

### 4. Feature Selection

- **Features (X)**: All columns except target variable
- **Target (y)**: Current_loan_status

### 5. Train-Test Split

- Split ratio: 80% training, 20% testing
- Random state: 42 for reproducibility

### 6. Model Training

```python
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
```

### 7. Model Evaluation

Generate predictions and evaluate using:

- Accuracy score
- Confusion matrix
- Visualization with heatmap

### 8. Model Persistence

Save the trained model for future use:

```python
import joblib
joblib.dump(model, 'customer_default_prediction_model.pkl')
```

## 📈 Results

### Model Performance

- **Accuracy**: **92.88%**

### Confusion Matrix Analysis

The model demonstrates strong predictive performance:

- **True Positives**: 5002 customers correctly predicted to default
- **True Negatives**: 1052 customers correctly predicted to not default
- **False Positives**: 324 customers incorrectly predicted to default
- **False Negatives**: 140 customers incorrectly predicted to not default

### Interpretation

The model shows excellent performance with over 92% accuracy. While there's room for improvement through hyperparameter tuning and feature engineering, the current model provides reliable predictions for loan default risk assessment.

## 🎨 Visualization

The project includes a confusion matrix heatmap that visualizes:

- Model predictions vs. actual outcomes
- Distribution of correct and incorrect classifications
- Clear distinction between default and non-default predictions

## 🔮 Future Improvements

1. **Hyperparameter Tuning**: Optimize n_estimators, learning_rate, and other parameters
2. **Feature Engineering**: Create new features from existing data
3. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
4. **Additional Metrics**: Include precision, recall, F1-score, and ROC-AUC
5. **Model Comparison**: Compare AdaBoost with other algorithms (Random Forest, XGBoost, etc.)
6. **Cost-Sensitive Learning**: Account for the different costs of false positives vs. false negatives
7. **Feature Importance Analysis**: Identify the most influential features

## 🌐 Web Application

This project includes a **fully functional web interface** for easy interaction with the prediction model!

### Features

- 🎨 **Modern & Responsive UI**: Beautiful gradient design that works on all devices
- 📊 **Real-time Predictions**: Instant loan default risk assessment
- 📈 **Visual Confidence Metrics**: Interactive bar charts showing prediction probabilities
- 💡 **Smart Recommendations**: Actionable insights based on risk levels
- ⚡ **Auto-calculations**: Automatic loan-to-income ratio computation

### Running the Web Application

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Train the Model (if not already done)

Run the Jupyter notebook to generate `customer_default_prediction_model.pkl`

#### 3. Start the Flask Server

```bash
python app.py
```

#### 4. Access the Application

Open your browser and navigate to:

```
http://localhost:5000
```

### Using the Web Interface

1. **Fill in the form** with customer and loan information:
   - Personal details (age, income, home ownership, employment)
   - Loan details (amount, purpose, grade, interest rate)
   - Credit history information

2. **Click "Predict Default Risk"** to get instant results

3. **View comprehensive results** including:
   - Default/No Default prediction
   - Risk level assessment
   - Confidence percentages for both outcomes
   - Personalized recommendations

### API Endpoints

The Flask backend provides the following endpoints:

- `GET /` - Main web interface
- `POST /predict` - Prediction API endpoint
  ```json
  {
    "customer_age": 35,
    "customer_income": 50000,
    "home_ownership": "RENT",
    "employment_duration": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 5.5,
    "loan_percent_income": 0.2,
    "historical_default": "N",
    "credit_history_length": 10
  }
  ```
- `GET /health` - Server health check

## 📁 Project Structure

```
Customer_Default_Prediction_using_AdaBoost(LOAN)/
│
├── Customer_Default_Prediction_using_AdaBoost(LOAN).ipynb  # Model training notebook
├── app.py                                                   # Flask backend server
├── requirements.txt                                         # Python dependencies
├── README.md                                                # Project documentation
├── customer_default_prediction_model.pkl                    # Trained model (generated)
│
├── templates/
│   └── index.html                                          # Web interface HTML
│
└── static/
    ├── style.css                                           # Styling and design
    └── script.js                                           # Frontend logic & API calls
```

## 🎓 Learning Outcomes

- Understanding ensemble learning techniques
- Implementing AdaBoost for binary classification
- Handling missing data and preprocessing steps
- Encoding categorical variables
- Model evaluation and interpretation
- Practical application of machine learning in finance

## 📝 Usage

To use the trained model for predictions:

```python
import joblib

# Load the saved model
model = joblib.load('customer_default_prediction_model.pkl')

# Make predictions on new data
predictions = model.predict(new_data)
```

## ⚠️ Important Notes

- Ensure all preprocessing steps applied to training data are also applied to new data
- The model requires numerical input, so categorical variables must be encoded
- Missing values should be imputed before making predictions

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

Created as part of a Machine Learning project portfolio.

## 🙏 Acknowledgments

- Dataset source: [GitHub Repository](https://github.com/itsluckysharma01/Datasets)
- scikit-learn documentation for AdaBoost implementation
- Machine Learning community for best practices

---

**Note**: This is a demonstration project for educational purposes. For production use in financial applications, additional validation, regulatory compliance, and risk assessment would be required.
