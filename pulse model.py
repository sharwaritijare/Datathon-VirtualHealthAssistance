import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Read the dataset
df = pd.read_csv("C:/Users/mailv/OneDrive/Desktop/sensor1.csv")

# Extract Pulse data for anomaly detection
pulse_data = df[['Pulse']]

# Scale the data using StandardScaler
scaler = StandardScaler()
pulse_scaled = scaler.fit_transform(pulse_data)

# Create the Isolation Forest model
model = IsolationForest(n_estimators=50, contamination=0.1, random_state=2)

# Fit the model to the scaled pulse data
model.fit(pulse_scaled)

# Make predictions (1 for normal, -1 for anomaly)
pulse_pred = model.predict(pulse_scaled)

# Add predictions (Anomaly label) to the dataframe
df['Pulse_Anomaly'] = pulse_pred

# Extract the anomalies
pulse_anomalies = df[df['Pulse_Anomaly'] == -1]

# Print the anomalies in the specified format
print("Detected Pulse Anomalies (Tabular Format):")
for index, row in pulse_anomalies.iterrows():
    print(f"Anomaly detected in pulse rate at datetime {row['Datetime']} with Pulse Rate = {row['Pulse']}")

# Plot the anomaly detection results
plt.figure(figsize=(10, 6))

# Define colors for normal and anomalous points
colors = df['Pulse_Anomaly'].map({1: 'blue', -1: 'red'})

# Scatter plot for Pulse vs. Anomaly detection results
plt.scatter(df['Datetime'], df['Pulse'], c=colors, label='Normal')

# Set title and labels
plt.title('Anomaly Detection in Pulse Data')
plt.xlabel('Datetime')
plt.ylabel('Pulse Rate')
plt.legend()

# Hide datetime values on the x-axis for clarity
plt.xticks([])

# Show the plot
plt.show()

# Save the model and scaler
joblib.dump(model, "pulse_anomaly_detection_model.pkl")
joblib.dump(scaler, "pulse_scaler.pkl")

# Optionally, save the anomalies to a CSV
pulse_anomalies.to_csv("pulse_anomalies.csv", index=False)
