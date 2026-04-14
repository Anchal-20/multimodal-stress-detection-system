import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

data = pd.read_excel('data/qt_dataset2.xlsx')
features = data[['Temperature', 'Oxygen', 'PulseRate']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled)

joblib.dump(kmeans, 'models/stress_kmeans_model.pkl')
joblib.dump(scaler, 'models/stress_scaler.pkl')

print("KMeans model saved!")
