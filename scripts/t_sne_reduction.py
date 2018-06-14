import os
import numpy as np
import pandas as pd
from os.path import join
from glob import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def tsne_reduction(x, labels, output_dir, dim=2, perplexity=50,
                   lr=50, n_iter=5000):
    """
    Reduce high dimension to 2D or 3D.
    Args:
        x: an np array (num_sample, num_feature)
        split: generate n number of tsne plots from the given samples, where
            each plot contain 'split' number of samples
    """

    # Use PCA to reduce dimensions to 50
    if x.shape[1] > 50:
        print("Using PCA to reduce dimension to 100.")
        pca = PCA(n_components=50)
        x = pca.fit_transform(x)

        print("After PCA, {:2f} of original variance remains.".format(
            np.sum(pca.explained_variance_ratio_)
        ))

    # t-sne reduction
    tsne = TSNE(n_components=dim,
                perplexity=perplexity,
                learning_rate=lr,
                n_iter=n_iter,
                verbose=1)

    x = tsne.fit_transform(x)

    df = pd.DataFrame(x, columns=['x', 'y', 'z'][:dim])
    df['label'] = labels

    # Save the df
    df.to_csv(join(output_dir, "tsne.csv"), index=False)


def generate_data(input_dir):
    """
    Generate x and labels for tsne_reduction.
    """

    features = []
    labels = []
    for sub in [f.path for f in os.scandir(input_dir)]:
        for npz in glob(join(sub, '*.npz')):
            loaded_npz = np.load(npz)
            features.append(loaded_npz['feature'])
            labels.append(loaded_npz['cpd'].tolist())

    # Concatenate the features into size (num_sample, num_feature)
    features = np.vstack(features)
    return features, labels


if __name__ == '__main__':
    x, labels = generate_data('./test')
    print("shape of x: {}".format(x.shape))
    print("There are {} compounds".format(len(set(labels))))
    tsne_reduction(x, labels, './tsnes', dim=2, n_iter=250)

