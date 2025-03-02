from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# loading data
df = pd.read_csv("C:/Users/mailv/OneDrive/Desktop/sensor1.csv")

# column selecction
X = df[['SYS', 'DIA', 'Pulse']]

# data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# isolation forest model
model1 = IsolationForest(contamination=0.05)

# fitting model
model1.fit(X_scaled)

# anomaly prediction
y_pred_if = model1.predict(X_scaled)

df['Anomaly_IF'] = y_pred_if

# filtering anomalies
anomalies_if = df[df['Anomaly_IF'] == -1]

print("Detected Anomalies using Isolation Forest:")
for index, row in anomalies_if.iterrows():
    print(f"Anomaly detected at datetime value of {row['Datetime']} with SYS = {row['SYS']} and DIA = {row['DIA']}")

plt.figure(figsize=(10, 6))
colors = df['Anomaly_IF'].map({1: 'blue', -1: 'red'})
plt.scatter(df['SYS'], df['DIA'], c=colors, label='Normal')
plt.title('Anomaly Detection in Blood Pressure Data')
plt.xlabel('Systolic Blood Pressure (SYS)')
plt.ylabel('Diastolic Blood Pressure (DIA)')
plt.legend()
plt.show()

anomalies_if.to_csv("anomalies.csv", index=False)

# save model
joblib.dump(model1, "if_bp.pkl")
joblib.dump(scaler, "scaler.pkl")
