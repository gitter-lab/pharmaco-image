import umap
import numpy as np
import pandas as pd
from os.path import join
from sklearn.decomposition import PCA


def umap_reduction(x, labels, output_dir, dim=2, n_neighbors=15,
                   label_2=None, label_3=None):
    """
    Reduce high dimension to 2D or 3D using umap.
    Args:
        x: an np array (num_sample, num_feature)
        labels: the categorical attribute of the rows in x
    """
    # Use PCA to reduce dimensions to 2048
    if x.shape[1] > 10000:
        print("Using PCA to reduce dimension to 2048.")
        pca = PCA(n_components=2048)
        x = pca.fit_transform(x)

        print("After PCA, {:2f} of original variance remains.".format(
            np.sum(pca.explained_variance_ratio_)
        ))

    # t-sne reduction
    x = umap.UMAP(n_neighbors=n_neighbors,
                  n_components=dim,
                  metric='cosine',
                  verbose=True,
                  n_epochs=1000).fit_transform(x)

    df = pd.DataFrame(x, columns=['x', 'y', 'z'][:dim])
    df['label'] = labels
    if label_2:
        df['label_2'] = label_2
    if label_3:
        df['label_3'] = label_3

    # Save the df
    df.to_csv(join(output_dir, "umap_{}d.csv".format(dim)), index=False)


x = np.load("./combined_feature.npz")['features']
labels = np.load("./combined_feature.npz")['names']
sids = np.load("./combined_feature.npz")['sids']
umap_reduction(x, labels, './', label_2=sids)
