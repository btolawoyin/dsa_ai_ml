from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import time

print("Current time:", datetime.now().strftime("%H:%M:%S"))
start_time = time.time()

# Load and check the shape of the original dataset
df_cleaned = pd.read_csv('train_transaction.csv')
print("COMPLETED: Load and check the shape of the dataset\n")

# Show shape and first few rows
print(f"Dataset shape: {df_cleaned.shape}")
print(df_cleaned.head())
print(f"💾 Approx. memory usage: {df_cleaned.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# See top memory-heavy columns
print("\nTop 10 memory-consuming columns:")
print(df_cleaned.memory_usage(deep=True).sort_values(ascending=False).head(10))

# Count missing values
missing = df_cleaned.isnull().sum()
missing_percent = (missing / len(df_cleaned)) * 100
missing_df = pd.DataFrame({'MissingValues': missing, 'Percent': missing_percent})
missing_df = missing_df[missing_df['MissingValues'] > 0].sort_values(by='Percent', ascending=False)
print("Show top 20 most missing")
print(missing_df.head(20))  # Show top 20 most missing
print("Check for missing values")
print("\nMissing values:\n", df_cleaned.isnull().sum()) # Check for missing values
print("COMPLETED: Count missing values\n")

# Class Distribution
print(df_cleaned['isFraud'].value_counts())
print(df_cleaned['isFraud'].value_counts(normalize=True))
print("COMPLETED: Class Distribution\n")

# Summary stats
print("\nSummary statistics:\n", df_cleaned.describe()) # Summary stats
print("COMPLETED: Summary stats")

# Drop columns with >90% missing
threshold = 90
cols_to_drop = missing_df[missing_df['Percent'] > threshold].index
df_cleaned = df_cleaned.drop(columns=cols_to_drop)
print(f"Dropped {len(cols_to_drop)} columns with >{threshold}% missing values.")
print("COMPLETED: Drop columns with >90% missing\n")

# Fill categorical with 'unknown', numeric with median
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col] = df_cleaned[col].fillna('unknown')
    else:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
print("COMPLETED: Fill categorical with 'unknown', numeric with median\n")

selector = VarianceThreshold(threshold=0.01) # Dropping Low-Variance or Constant Columns
df_reduced = selector.fit_transform(df_cleaned.select_dtypes(include=['number']))

# Reusable Function to Save Plot with Title and Timestamp
def save_plot(title, folder='plots'):
    """
    Saves the current matplotlib plot with a title and timestamp.
    Parameters:
    - title: str — The title for the plot (used in filename and chart)
    - folder: str — The folder to save plots into
    """
    os.makedirs(folder, exist_ok=True) # Create the folder if it doesn’t exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Timestamp
    filename = f"{title.replace(' ', '_')}_{timestamp}.png" # Clean filename
    plt.title(title) # Set plot title
    filepath = os.path.join(folder, filename) # Save plot
    plt.savefig(filepath, bbox_inches='tight')
    print(f"✅ Plot saved: {filepath}")

# Visualise Key Features

# Plot class distribution
sns.countplot(x='isFraud', data=df_cleaned)
plt.title("Fraud vs Legit Transactions")
#plt.show()
plt.suptitle("Fraud vs Legit Transactions")
save_plot("Fraud vs Legit Transactions")

# Compare transaction amount distribution
sns.boxplot(x='isFraud', y='TransactionAmt', data=df_cleaned)
plt.title("Transaction Amount by Class")
#plt.show()
plt.suptitle("Transaction Amount by Class")
save_plot("Transaction Amount by Class")
print("COMPLETED: Plotting of Transaction Amount by Class\n")

# Distribution of 'Transaction Amount' and 'Time'
print(df_cleaned.columns.tolist()) # Print exact column names present in the Dataset

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_cleaned['TransactionAmt'], bins=50, ax=axs[0])
axs[0].set_title("Transaction Amount Distribution")
#plt.show()
plt.suptitle("Distribution of Transaction Amounts")
save_plot("Distribution of Transaction Amounts")
print("COMPLETED: Plotting of Distribution of Transaction Amounts\n")

sns.histplot(df_cleaned['TransactionDT'], bins=50, ax=axs[1]) # Plot histogram
axs[1].set_title("Transaction Time Distribution")

plt.tight_layout()
#plt.show()
plt.suptitle("Transaction Time Distribution")
save_plot("Transaction Time Distribution")
print("COMPLETED: Plotting of Transaction Time Distribution\n")

# Class imbalance barplot
sns.countplot(x='isFraud', data=df_cleaned)
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
#plt.show()
plt.suptitle("Class Distribution")
save_plot("Class Distribution")
print("COMPLETED: Visualise Key Features\n")
print("COMPLETED: Plotting of Visualise Key Features\n")

# Check Class Imbalance
counter = Counter(df_cleaned['isFraud'])
print(f"Legit: {counter[0]} ({counter[0]/len(df_cleaned)*100:.4f}%)")
print(f"Fraud: {counter[1]} ({counter[1]/len(df_cleaned)*100:.4f}%)")
print("COMPLETED: Check Class Imbalance\n")

# Drop unnecessary columns (e.g. TransactionID)
if 'TransactionID' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['TransactionID'])
print("COMPLETED: Drop unnecessary columns (TransactionID)\n")    

# Drop duplicates
before = df_cleaned.shape[0]
df_cleaned = df_cleaned.drop_duplicates()
after = df_cleaned.shape[0]
print(f"Removed {before - after} duplicate rows.")
print("COMPLETED: Drop duplicates\n")    

# Optionally reset index
df_cleaned.reset_index(drop=True, inplace=True)
print("COMPLETED: reset index\n")    

# Separate features and target
X = df_cleaned.drop(['isFraud'], axis=1)
y = df_cleaned['isFraud']
print("COMPLETED: Separate features and target\n")

# Separate numeric and categorical columns
X_numeric = X.select_dtypes(include=['int64', 'float64'])
print("COMPLETED: Separate numeric and categorical columns\n")

# Fill missing values in numeric columns
X_numeric = X_numeric.fillna(0)
print("COMPLETED: Fill missing values in numeric columns\n")

# Save the column names used for training (before scaling)
training_columns = X_numeric.columns.tolist()

# Scale only numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
print("COMPLETED: Scale only numeric features\n")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("COMPLETED: Train-test split\n")

# Save column names BEFORE transformations
feature_names = training_columns
print("COMPLETED: Stored the exact columns BEFORE scaling\n")

# Apply SMOTE to training set
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Shapes after SMOTE:")
print("X_train_sm:", X_train_sm.shape)
print("y_train_sm:", y_train_sm.shape)
print("Apply SMOTE to training set\n")

# Convert back to DataFrame
X_train = pd.DataFrame(X_train, columns=feature_names)

final_columns = X_train.columns.tolist()

# Model Development
# Supervised ML algorithm: Logistic Regression
start1 = timer()

# Dropping Low-Variance or Constant Columns
# selector = VarianceThreshold(threshold=0.01)
# X_reduced = selector.fit_transform(X_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print("COMPLETED: Supervised ML algorithm: Logistic Regression")
end1 = timer()
print(f"⏱️ Logistic Regression Section took: {end1 - start1:.4f} seconds")

# Random Forest
start2 = timer()

# Limiting SMOTE Volume
# Using a sample/applying SMOTE only on a reduced training set (say 50%):

rf = RandomForestClassifier(
    n_estimators=50,         # Fewer trees
    max_depth=10,            # Limit tree depth
    n_jobs=-1,               # Use all CPU cores
    random_state=42
)

rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)
print("Random Forest Report:\n")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print("COMPLETED: Supervised ML algorithm: Random Forest")
end2 = timer()
print(f"⏱️ Random Forest Section took: {end2 - start2:.4f} seconds")

# XGBoost (Extreme Gradient Boosting)
start3 = timer()

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_sm, y_train_sm)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost Report:\n")
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print("COMPLETED: Supervised ML algorithm: XGBoost (Extreme Gradient Boosting)")
end3 = timer()
print(f"⏱️ XGBoost Section took: {end3 - start3:.4f} seconds")

# ========== DEPRECATED ========== 
# Code block left for historical purposes
# Neural Network (MLP Classifier from Scikit-learn)
# start4 = timer()
# mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
# mlp.fit(X_train_sm, y_train_sm)
# y_pred_mlp = mlp.predict(X_test)

# print("Neural Network (MLP) Report:\n")
# print(classification_report(y_test, y_pred_mlp))
# print(confusion_matrix(y_test, y_pred_mlp))
# print("COMPLETED: Supervised ML algorithm: Neural Network (MLP Classifier from Scikit-learn)")
# end4 = timer()
# print(f"⏱️ Neural Network Section took: {end4 - start4:.4f} seconds")
# ========== DEPRECATED ========== 

# ROC AUC Comparison
print("ROC-AUC Scores:")
print("Logistic Regression:", roc_auc_score(y_test, y_pred_lr))
print("Random Forest:", roc_auc_score(y_test, y_pred_rf))
print("XGBoost:", roc_auc_score(y_test, y_pred_xgb))
print("COMPLETED: Supervised ML algorithm: ROC AUC Comparison\n")

#print("COMPLETED: \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏳ Execution started at: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
print("Current time:", datetime.now().strftime("%H:%M:%S"))
print(f"⏳ Execution Time: {elapsed_time:.2f} seconds")

# Save model and scaler to file
with open("model.pkl", "wb") as model_file:
    pickle.dump(xgb, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and scaler saved as model.pkl and scaler.pkl")

# Save the exact training columns (after preprocessing!)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(training_columns, f)

print("COMPLETED: =====END OF SCRIPT=====\n")
