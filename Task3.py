# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2. Load the CSV file
df = pd.read_csv('Housing.csv')  # Replace with your actual file name
print(df.head)

# 3. Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# 4. Identify categorical and numerical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# 5. Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Keep numerical features as-is
)

# 6. Create a pipeline with preprocessing + Linear Regression
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Plot Actual vs Predicted Prices
plt.scatter(y_test, y_pred, color='green', edgecolor='k')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# 11. Coefficients interpretation
regressor = model.named_steps['regressor']
preprocessor_fit = model.named_steps['preprocessing']
feature_names = preprocessor_fit.get_feature_names_out()
coef = regressor.coef_


print("\n🔍 Model Coefficients:")
for name, c in zip(feature_names, coef):
    print(f"{name}: {c:.2f}")

print(f"Intercept: {regressor.intercept_:.2f}")
