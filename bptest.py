import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load model
loaded_model = joblib.load("if_bp.pkl")
loaded_scaler = joblib.load("scaler.pkl")

new_data = pd.DataFrame({
    'Datetime': ['07-05-2021  11:05:00', '13-05-2021  10:00:00', '25-06-2021  08:20:00', '03-07-2021  08:34:00', '10-07-2021  21:19:00', '21-07-2021  11:57:00', '13-09-2021  10:24:00'],
    'SYS': [120, 125, 130, 156, 120, 71, 178],
    'DIA': [80, 85, 90, 98, 80, 50, 96],
    'Pulse': [72, 78, 80, 73, 100, 62, 111]
})

X_new = new_data[['SYS', 'DIA', 'Pulse']]
X_new_scaled = loaded_scaler.transform(X_new)

y_pred_new = loaded_model.predict(X_new_scaled)
new_data['Anomaly_IF'] = y_pred_new

print("Detected Anomalies in New Data:")
for index, row in new_data.iterrows():
    if row['Anomaly_IF'] == -1:  
        print(f"Anomaly detected at datetime value of {row['Datetime']} with SYS = {row['SYS']} and DIA = {row['DIA']}")

# plotting anomalies
plt.figure(figsize=(10, 6))

colors = new_data['Anomaly_IF'].map({1: 'blue', -1: 'red'})

plt.scatter(new_data['SYS'], new_data['DIA'], c=colors, label='Normal')

plt.scatter(new_data[new_data['Anomaly_IF'] == -1]['SYS'],
            new_data[new_data['Anomaly_IF'] == -1]['DIA'],
            color='red', label='Anomaly')

plt.title('Anomaly Detection in New Data (SYS vs DIA)')
plt.xlabel('Systolic Blood Pressure (SYS)')
plt.ylabel('Diastolic Blood Pressure (DIA)')
plt.legend()

plt.savefig("C:/Users/mailv/OneDrive/Pictures/anomalies/analysis-graph.png")
print('Image saved')

plt.show()

new_data.to_csv("new_data_with_anomalies.csv", index=False)
