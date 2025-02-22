import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("/content/traffic.csv")

# Convert DateTime column to proper format
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime")

# Select a single junction for clustering
junction_id = 1
df_junction = df[df["Junction"] == junction_id][["DateTime", "Vehicles"]]

# Extract vehicle data for clustering
X = df_junction[["Vehicles"]].values

# Finding optimal K using Elbow Method
wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

# Plot Elbow Method Graph
plt.plot(range(1, 11), wcss_list)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# Applying K-Means with the optimal K (Assuming 3 for traffic levels)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(X)

# Visualizing Clusters
plt.scatter(X[y_predict == 0], df_junction["DateTime"][y_predict == 0], s=50, c='blue', label='Low Traffic')
plt.scatter(X[y_predict == 1], df_junction["DateTime"][y_predict == 1], s=50, c='green', label='Moderate Traffic')
plt.scatter(X[y_predict == 2], df_junction["DateTime"][y_predict == 2], s=50, c='red', label='High Traffic')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], [df_junction["DateTime"].median()] * 3,
            s=200, c='yellow', label='Centroids')

plt.title('Traffic Clusters')
plt.xlabel('Vehicle Count')
plt.ylabel('Time')
plt.legend()
plt.show()
