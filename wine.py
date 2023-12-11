import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings("ignore")

# Table(order)
df = pd.read_csv("wine.csv")
df.head()

# List
df.info()

# Reverse rows and columns
df.describe().T

# EDA
# Histogram
df.hist(figsize=(15, 15))
plt.show()

# Box-plot
df.plot(kind="box", subplots = True, layout = (4, 4), figsize = (15, 15))
plt.show()

# Heat-map
plt.figure(figsize = (15, 10))
mask = np.triu(df.corr(), 1)
sns.heatmap(df.corr(), annot = True, mask = mask, cmap = "crest")
plt.show()

# Processing
# Sum of Null Values
df.isnull().sum()

# Sum of Duplicated Value
df.duplicated().sum()

# Outlier Detection
print("Before outlier Detection:", df.shape)

for col in df.columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    IQR = q3 - q1
    
    max_val = q3 + 1.5 * IQR
    min_val = q1 - 1.5 * IQR
    
    outliers = df[(df[col]>max_val) | (df[col]<min_val)].index
    
    df.drop(outliers, axis = 0, inplace = True)

print("After outlier Detection:", df.shape)

# Skewness
df.skew().sort_values(ascending = False)

# Scaling 1
columns = df.columns
scaler = StandardScaler()

data = scaler.fit_transform(df)

df = pd.DataFrame(data = data, columns = columns)
df.head()

# Modeling
# PCA
pca = PCA(n_components = 2)
pca_2 = pca.fit_transform(data)

# Scatter-plot
# 13 Dimensions to 2 dimensions
plt.figure(figsize = (12, 8))
sns.scatterplot(x = pca_2[:, 0], y = pca_2[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# K-Means
# K-Means Inertia and Silhouette score
inertia = list()
silhouette = {}

for i in range(2, 10):
    k_means = KMeans(n_clusters = i)
    k_means.fit(data)
    
    inertia.append(k_means.inertia_)
    
    silhouette[i] = silhouette_score(df, labels = k_means.labels_, metric = "euclidean")
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))

# Line-plot
sns.lineplot(x = range(2, 10), y = inertia, marker = "o", ax = ax[0])
ax[0].set_xlabel("Number of Cluster")
ax[0].set_ylabel("Inertia")
ax[0].set_title("Inertia by Number of Cluster")

# Bar-plot
sns.barplot(x = list(silhouette.keys()), y = list(silhouette.values()), ax = ax[1])
ax[1].set_title("Silhouette Score by Number of Cluster")
ax[1].set_xlabel("Number of Cluster")
ax[1].set_ylabel("Silhouette Score")
plt.show()

# In 'K = 3' all the metrics indicates that it is the best clusters number
kmeans = KMeans(n_clusters = 3)
kmeans.fit(data)

kmeans_labels = kmeans.predict(data)
centers = kmeans.cluster_centers_

pca = PCA(n_components = 2)
centers = pca.fit_transform(centers)

centers

# Scatter
plt.figure(figsize = (12, 8))
plt.scatter(pca_2[:, 0], pca_2[:, 1], c = kmeans_labels, cmap = "Paired")
plt.scatter(centers[:, 0], centers[:, 1], c = "red", s = 200)
plt.show()