import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load your customer data
df = pd.read_csv("Online_Retail_Cleaned.csv")

# Select numerical features for clustering
X = df.select_dtypes(include=['int64', 'float64'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train clustering model (adjust n_clusters to your project)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save models
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("✅ Models retrained and saved successfully")
