# Machine-fault-detection
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training data
data = {
    "Temperature": [40, 45, 50, 70, 75, 80, 42, 48, 72, 78],
    "Vibration":   [2, 3, 2, 8, 9, 10, 3, 2, 8, 9],
    "Current":     [5, 5, 6, 9, 10, 11, 5, 6, 9, 10],
    "Fault":       [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Input features
X = df[["Temperature", "Vibration", "Current"]]

# Output
y = df["Fault"]

# Create AI model
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Get machine sensor values
temperature = float(input("Enter temperature (°C): "))
vibration = float(input("Enter vibration value: "))
current = float(input("Enter current (A): "))

# Predict fault
prediction = model.predict([[temperature, vibration, current]])

# Display result
if prediction[0] == 1:
    print("\n⚠ Machine Fault Detected!")
    print("Maintenance is required.")
else:
    print("\n✓ Machine is operating normally.")
