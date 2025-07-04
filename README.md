# AI-Powered Fraud Detection System

## Project Overview

**Project Question**  
How can AI models be developed and deployed in real-world applications to support decision-making and improve task automation through a simple web interface?

This is a Flask-based machine learning application that detects fraudulent transactions using supervised learning algorithms. Users input transaction data through a simple web interface, and the app predicts whether the transaction is legitimate or fraudulent.

**Use Case**  
Fraud Detection – Identifying fraudulent transactions using payment data patterns.

---

## Project Structure

```
dsa-project/
├── app.py                        # Flask web app
├── train_and_save_model.py      # Model training script
├── scaler.pkl                   # Saved scaler
├── model.pkl                    # Trained ML model
├── feature_columns.pkl          # Feature list used for inference
├── smote_balanced_data.csv      # Cleaned and balanced dataset
├── prediction_log.csv           # Logs user inputs and predictions
├── requirements.txt             # Python package dependencies
├── runtime.txt                  # Python version
├── templates/
│   ├── index.html               # Input form
│   └── result.html              # Output page
```

---

## Model Summary

The following supervised ML models were trained and evaluated:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Neural Network (MLP Classifier)

**Performance Metrics**  
- Precision  
- Recall  
- F1 Score  
- ROC AUC  

| Model              | Precision | Recall | F1 Score | ROC AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.91      | 0.87   | 0.89     | 0.92    |
| Random Forest       | 0.96      | 0.93   | 0.94     | 0.98    |
| XGBoost             | 0.97      | 0.94   | 0.95     | 0.99    |
| Neural Network      | 0.96      | 0.92   | 0.94     | 0.97    |

> Models were trained on SMOTE-balanced data to improve detection of fraud cases.

---

## Data & Resources

**Datasets Available:**
- `train_transaction.csv` – Raw dataset  
- `smote_balanced_data.csv` – SMOTE-balanced dataset  
- `model.pkl` – Trained model  
- `scaler.pkl` – Feature scaler  
- `feature_columns.pkl` – Selected input features  
- Flask app files: `app.py`, `templates/`, etc.

> Note: Large datasets are hosted externally due to GitHub file size limitations.  
🔗 [Download Datasets & Resources](https://bit.ly/dsa_ai_ml)

---

## Setup Instructions (Windows 11 – VS Code)

```bash
# Clone the repository
git clone https://github.com/btolawoyin/dsa_ai_ml.git
cd dsa_ai_ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Flask Web App

```bash
python app.py
```

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.  
You'll see a form to enter transaction data and receive a fraud prediction.

---

## Deployment to Render

**Project deployed:**  
🔗 [Live App](https://ai-powered-fraud-detection-system.onrender.com)

**Steps to Deploy:**
1. Push code to GitHub.
2. Log in at [Render Dashboard](https://dashboard.render.com).
3. Click “New Web Service” and link your GitHub repo.
4. Set Start Command to:
   ```bash
   gunicorn app:app
   ```
5. Select ‘Free Instance Type’ and click ‘Create Web Service’.

**Note:**  
If you get `gunicorn: command not found`, make sure it's added:

```bash
pip install gunicorn
pip freeze > requirements.txt
```

Then commit and push the updated `requirements.txt` to GitHub.

---

## Reflection

**What Worked**
- SMOTE helped with class imbalance.
- XGBoost gave the best performance.
- Flask UI enabled easy interaction.

**What Didn’t**
- Neural Networks were slow to train.
- GitHub blocked large file uploads.

**Future Improvements**
- Real-time fraud detection with streaming APIs.
- Docker & Kubernetes orchestration.
- Alerts for high-risk transactions.

---

## Tools Used

**Language:** Python 3.10  
**Libraries:**  
- pandas, numpy, scikit-learn  
- XGBoost, imbalanced-learn  
- Flask, seaborn, matplotlib  

**Dev Tools:**  
- Git, GitHub  
- Visual Studio Code (Windows 11)

---

## Testing

Sample test inputs (5 sets) are built into the form. Custom values can be tested as well. Predictions and probabilities are logged in `prediction_log.csv`.

---

## License

> This project is for educational and non-commercial use only.  
> Models must be validated before production use.

---

## Author

**Name:** **Bukky Olawoyin**  
Cybersecurity & Forensics Researcher | Software Engineer | Agricultural Investor  
🔗 [LinkedIn](https://www.linkedin.com/in/bukkyolawoyin/)
**Project:** DSA Projects – Artificial Intelligence
