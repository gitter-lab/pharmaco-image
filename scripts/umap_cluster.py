import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Load the UMAP features
df = pd.read_csv("./umap_2d.csv")
x = []
for index, row in df.iterrows():
    x.append([float(row['x']), float(row['y'])])
x = np.vstack(x)

# Hierarchical clustering
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
                              linkage='average')
agg.fit(x)

# Save the label with other features
df['cluster'] = agg.labels_

pd.to_csv("./umap_2d_cluster.csv")
