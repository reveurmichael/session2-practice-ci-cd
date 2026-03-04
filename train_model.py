import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load training data
data = pd.read_csv("training_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train linear model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "linear_model.pkl")

# Save model coefficients to text file
with open("linear_model.txt", "w") as f:
    f.write(f"Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n")

