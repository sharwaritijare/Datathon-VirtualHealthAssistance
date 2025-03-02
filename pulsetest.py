import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load the saved model and scaler
model = joblib.load("pulse_anomaly_detection_model.pkl")
scaler = joblib.load("pulse_scaler.pkl")

# Assuming you have new data in a CSV file or as a DataFrame
# Example: new_data.csv (make sure to provide the correct file path)
new_data = pd.DataFrame({'Datetime': ['07-05-2021  11:05:00', '13-05-2021  10:00:00', '25-06-2021  08:20:00', '03-07-2021  08:34:00', '10-07-2021  21:19:00', '21-07-2021  11:57:00', '13-09-2021  10:24:00'],'Pulse': [72, 78, 80, 73, 100, 62, 111]})

# Extract Pulse data from the new data
new_pulse_data = new_data[['Pulse']]

# Standardize the new data using the saved scaler
new_pulse_scaled = scaler.transform(new_pulse_data)

# Predict anomalies using the loaded model
new_pulse_pred = model.predict(new_pulse_scaled)

# Add the predictions (Anomaly labels) to the new data
new_data['Pulse_Anomaly'] = new_pulse_pred

# Extract the anomalies
new_pulse_anomalies = new_data[new_data['Pulse_Anomaly'] == -1]

# Print detected anomalies (formatted output)
print("Detected Pulse Anomalies in New Data (Tabular Format):")
for index, row in new_pulse_anomalies.iterrows():
    print(f"Anomaly detected in pulse rate at datetime {row['Datetime']} with Pulse Rate = {row['Pulse']}")

# Plot the anomaly detection results for the new data
plt.figure(figsize=(10, 6))

# Define colors for normal and anomalous points
colors = new_data['Pulse_Anomaly'].map({1: 'blue', -1: 'red'})

# Scatter plot for Pulse vs. Anomaly detection results
plt.scatter(new_data['Datetime'], new_data['Pulse'], c=colors, label='Normal')

# Set title and labels
plt.title('Anomaly Detection in New Pulse Data')
plt.xlabel('Datetime')
plt.ylabel('Pulse Rate')
plt.legend()

# Optionally, hide datetime values on the x-axis for clarity
plt.xticks(rotation=45)

# Save the graph as an image file
plt.savefig("C:/Users/mailv/OneDrive/Pictures/pulseano/pulse-analysis-graph")

# Show the plot (optional)
plt.show()

# Optionally, save the new anomalies to a CSV
new_pulse_anomalies.to_csv("new_pulse_anomalies.csv", index=False)
