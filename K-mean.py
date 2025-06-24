import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.txt", header=None, names=["longitude", "latitude"])

# Compute the mean of clusters
def compute_mean(points):
    # points: list of (longitude, latitude)
    return np.mean(points, axis=0)

def random_centroid(points, k):
    # Convert points to numpy
    points = np.asarray(points)
    initial_indices = np.random.choice(len(points), size=k, replace=False)
    centroid = points[initial_indices]
    return centroid

def compute_new_centroids(points, assignments, k):
    """
    Recompute centroids as the mean of all points assigned to each cluster.

    Parameters:
        points (np.ndarray): shape (n_points, 2), input coordinates
        assignments (list or np.ndarray): cluster labels for each point
        k (int): number of clusters

    Returns:
        np.ndarray: new centroids, shape (k, 2)
    """
    centroids = []
    for cluster_id in range(k):
        cluster_points = points[np.array(assignments) == cluster_id]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = points[np.random.randint(len(points))]
        centroids.append(centroid)
    return np.array(centroids)

## Step 1: Random Select K Centroids 

# K = 3, with 3 initial centroids
k_centroid = random_centroid(data, k=3)

## Step 2: Assign Points to K Clusters

# We use Euclidean Distance to get the determine the the closest centroid
# sqrt(d1^2 + d2^2)

def assign_clusters(points, centroids):
    """
    Assign each point to the index of the nearest centroid.

    Parameters:
        points: np.ndarray of shape (n_points, 2)
        centroids: np.ndarray of shape (k, 2)

    Returns:
        List of cluster indices (length = n_points)
    """
    
    clusters = []
    
    points = np.asarray(points)
    
    for p in points:
        # Normalization the distances
        distances = np.linalg.norm(centroids - p, axis=1)  # shape (k,)
        # Calculate which centroid is closest to the point
        closest = np.argmin(distances)
        clusters.append(closest)
    return clusters


## Step 3: Converegence Criterion
threshold = 1e-6  # convergence tolerance
max_iter = 100

k_centroid = random_centroid(data, k=3)  # initial centroids

for i in range(max_iter):
    cluster = assign_clusters(data, k_centroid)
    new_centroids = compute_new_centroids(data, cluster, k=3)

    # Compute how much centroids moved
    shift = np.linalg.norm(new_centroids - k_centroid)

    if shift < threshold:
        print(f"Converged at iteration {i}, centroid shift: {shift:.9f}")
        break

    k_centroid = new_centroids  # update for next iteration

else:
    print("Did not converge within max_iter")

# In to a data frame
df = data.copy()
df["cluster"] = cluster


# Plot the clusters

plt.figure(figsize=(8, 6))

# Plot each cluster separately for labeled legend
for cluster_id in sorted(df["cluster"].unique()):
    cluster_points = df[df["cluster"] == cluster_id]
    plt.scatter(
        cluster_points["longitude"],
        cluster_points["latitude"],
        marker='x',
        s=50,
        label=f"Cluster {cluster_id}"
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("K-means Clustering of Locations")
plt.legend()
plt.grid(True)
plt.show()

