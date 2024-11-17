# Credit Card Default Detection hi hghyiji

## 📊 Project Overview
This project implements a machine learning model to predict credit card default probability. It uses a Random Forest Classifier to analyze various customer attributes and payment patterns to assess the likelihood of credit default.

## 💻 Requirements

1. tensorflow
2. pandas
3. numpy
4. scikit-learn
5. tensorboard
6. matplotlib
7. streamlit
8. scikeras

## Installation
To run the project locally, follow these steps:

1. Clone the repository:

```cmd
git clone https://github.com/Yuvraj0014/Credit-Card-Fraud-Detection.git
Credit-Card-Fraud-Detection
```

2. Setup a virtual environment (optional but recommended)
```cmd
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate  # For Windows
```

3. Install required dependencies
```cmd
pip install -r requirements.txt
```

4. Run the streamlit app
```cmd
streamlit run app.py
```

## Results 
Too lazy to adult today? Join the club! Ditch the to-do list and dive into the Streamlight app instead. It's right here, waiting for you!
```
simple-credit-card-fraud-detection.streamlit.app
```



## 🚀 Features
The application considers multiple features across different categories to make predictions:

### 1. Personal Information
| Feature | Description | Values |
|---------|-------------|--------|
| LIMIT_BAL | Credit limit amount | NT dollars |
| SEX | Gender | 1 = male, 2 = female |
| EDUCATION | Education level | 1 = graduate school, 2 = university, 3 = high school, 4 = others |
| MARRIAGE | Marital status | 1 = married, 2 = single, 3 = others |
| AGE | Age in years | Numeric |

### 2. Payment Status History (PAY_1 to PAY_6)
Represents repayment status for the last 6 months (PAY_1 = last month, PAY_6 = 6 months ago)

| Value | Description |
|-------|-------------|
| -2 | No consumption |
| -1 | Paid in full |
| 0 | Revolving credit |
| 1 | Payment delay for one month |
| 2 | Payment delay for two months |
| 3+ | Payment delay for three or more months |

### 3. Bill Amounts (BILL_AMT1 to BILL_AMT6)
- Monthly bill statements for last 6 months
- BILL_AMT1 is the most recent
- Indicates spending patterns and credit utilization

### 4. Payment Amounts (PAY_AMT1 to PAY_AMT6)
- Amount paid for previous bills
- PAY_AMT1 is the most recent payment
- Shows payment behavior and ability to pay

## 💡 How It Works

### Data Preprocessing
```python
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- All numerical features are standardized to mean=0 and variance=1
- Ensures all features contribute equally to the model

### Model Architecture
- Uses Random Forest Classifier with 100 trees
- Ensemble learning approach for better generalization
- Each tree votes on the final prediction

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
```

### Risk Assessment
The model provides:
- Default probability (0-100%)
- Risk categorization:
  * 🟢 Low Risk: <30% default probability
  * 🟡 Medium Risk: 30-70% default probability
  * 🔴 High Risk: >70% default probability

### Key Risk Factors

1. **Payment History Impact**
   - Recent payment delays
   - Multiple months of delayed payments
   - Frequency of defaults

2. **Credit Utilization Patterns**
   - Ratio of bill amounts to credit limit
   - Trend in monthly bills
   - Available credit usage

3. **Payment Behavior**
   - Payment amounts vs. bill amounts
   - Consistency in payments
   - Payment timing patterns

## 🛠 Technical Implementation

### Requirements
```
streamlit
pandas
scikit-learn
numpy
```

### Installation & Setup
1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

### File Structure
```
├── README.md
├── app.py
├── requirements.txt
├── creditCardFraud_28011964_120214.csv
└── .gitignore
```

## 📊 Web Interface Features

1. **Input Sections**
   - Personal Information
   - Payment Status History
   - Bill and Payment Amounts

2. **Prediction Results**
   - Default Probability
   - Non-Default Probability
   - Risk Level Assessment

3. **Key Insights**
   - Automatic risk factor identification
   - Payment behavior analysis
   - Credit utilization warnings

## 🚨 Important Notes

1. **Data Privacy**
   - No personal information is stored
   - All predictions are made in real-time

2. **Model Limitations**
   - Based on historical data patterns
   - Should be used as a tool, not sole decision maker
   - Regular retraining recommended

3. **Best Practices**
   - Regular model validation
   - Monitor prediction accuracy
   - Update feature importance periodically

## Output Explanation
![image](https://github.com/user-attachments/assets/3982e3e3-c109-4ef9-8d76-53ddddf4638b)
![image](https://github.com/user-attachments/assets/3716720d-0bb5-4ac3-ad12-6e26504c7754)

---
*Note: This project is for educational purposes and should not be used as the sole decision-maker for credit evaluation.*
