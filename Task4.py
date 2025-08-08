# ------------------------------
# Breast Cancer Binary Classifier - Logistic Regression
# ------------------------------

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
# Change this to your CSV file path
df = pd.read_csv("breast_cancer_data.csv")

# 3. Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())

# 4. Handle missing values if any
df = df.dropna()

# 5. Encode categorical target if necessary
# Assume target column is 'diagnosis' where M = malignant, B = benign
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 6. Split features (X) and target (y)
X = df.drop(columns=['diagnosis', 'id'])  # remove target + any non-feature columns
y = df['diagnosis']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 10. Predictions
y_pred = model.predict(X_test)

# 11. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
