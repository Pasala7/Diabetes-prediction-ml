# Diabetes Prediction using Machine Learning
# Complete code in a single file

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Machine Learning Project: Diabetes Prediction")
print("=" * 50)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict for a sample patient
sample_patient = X_test[0].reshape(1, -1)
prediction = model.predict(sample_patient)

print("\nSample Prediction")
print("=" * 50)
print(f"Actual Disease Progression Score: {y_test[0]:.2f}")
print(f"Predicted Disease Progression Score: {prediction[0]:.2f}")

# Feature importance
print("\nFeature Importances")
print("=" * 50)
for i, importance in enumerate(model.feature_importances_):
    print(f"Feature {i+1}: {importance:.4f}")
