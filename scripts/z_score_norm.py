import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

sns.set(color_codes=True)


def group_features(features, names):
    """
    Group the features by plate list.
    Return two dictionaries: feature dict and DMSO feature dict
    """
    pids = set()
    for name in names:
        pids.add(int(re.sub(r'(\d{5})_.+', r'\1', name)))

    # Group the features by pid and DMSO
    dmso_feature_dict = {}
    feature_dict = {}
    for p in pids:
        dmso_feature_dict[p] = []
        feature_dict[p] = []

    # Add those features in the dictionary
    for i in range(len(names)):
        pid = int(re.sub(r'(\d{5})_.+', r'\1', names[i]))
        if "DMSO" in names[i]:
            dmso_feature_dict[pid].append(features[i, :])
        feature_dict[pid].append(features[i, :])

    return feature_dict, dmso_feature_dict


def compute_stats(feature_dict, dmso_feature_dict, names):
    """
    Compute the statistics used for later normalization. The first priority is
    to use the mean and std from DMSO images. If the std is 0 (rarely), we use
    the mean and std from all images. If the new std is also 0 (very rarely),
    we just give up normalizing that feature in that plate.
    """
    # Then we want to aggregate the features within each plate
    pids = set()
    for name in names:
        pids.add(int(re.sub(r'(\d{5})_.+', r'\1', name)))

    dmso_mean_dict = {}
    dmso_sd_dict = {}
    problem_sd_dict = {}

    for p in pids:
        # Use the DMSO mean and std
        dmso_mean_dict[p] = np.mean(dmso_feature_dict[p], axis=0)
        dmso_sd_dict[p] = np.std(dmso_feature_dict[p], axis=0)

        # Use the total image mean and std
        if 0 in dmso_sd_dict[p]:
            for col in np.where(dmso_sd_dict[p] == 0)[0]:
                print(col)
                dmso_mean_dict[p][col] = np.mean(
                    np.vstack(feature_dict[p])[:, col])
                dmso_sd_dict[p][col] = np.std(
                    np.vstack(feature_dict[p])[:, col])

                # If the sd of all images is still 0, then we just don't
                # normalize this feature and add it to a special list
                if np.std(np.vstack(feature_dict[p])[:, col]) == 0:
                    dmso_mean_dict[p][col] = 0
                    dmso_sd_dict[p][col] = 1
                    print(p, col)
                    if p in problem_sd_dict:
                        problem_sd_dict[p].append(col)
                    else:
                        problem_sd_dict[p] = [col]

    # Save the results
    np.savez("z_score_norm_dicts.npz", dmso_mean_dict=dmso_mean_dict,
             dmso_sd_dict=dmso_sd_dict, problem_sd_dict=problem_sd_dict)


def z_score_norm(feature, name, mean_dict, sd_dict):
    """
    Use z-score to normalize the given feature.
    Args:
        feature: an np.array of the current feature you want to normalize
        name: the associated name with this feature, expect something,
              like "24772_m06"
        mean_dict: a dictionary {pid: normalizing sample mean (a np.array)}
        sd_dict: a dictionary {pid: normalizing sample sd (a np.array)}
    """
    # Get the pid for this feature
    pid = int(re.sub(r'(\d{5})_.+', r'\1', name))
    return np.divide((feature - mean_dict[pid]), sd_dict[pid])


def vis_batch_effect(dmso_feature_dict, out_name="correlation.png"):
    """
    Visualize the median of dmso images across all plates.
    """
    # It is recommended to use median to aggregate features
    dmso_feature_median = []

    for p in dmso_feature_dict:
        cur_median = np.median(np.vstack(dmso_feature_dict[p]), axis=0)
        dmso_feature_median.append((p, cur_median))

    # Sort before plotting
    feature_median_sorted = sorted(dmso_feature_median, key=lambda x: x[0])

    sorted_dmso_median_feature = np.vstack(
        [i[1] for i in feature_median_sorted]
    )
    sorted_dmso_median_pid = [i[0] for i in feature_median_sorted]

    dmso_feature_cor = np.corrcoef(sorted_dmso_median_feature, rowvar=True)

    # Create a df for plotting
    df = pd.DataFrame(dmso_feature_cor)
    df.columns = sorted_dmso_median_pid
    df.index = sorted_dmso_median_pid

    fig, ax = plt.subplots(figsize=(30, 30))
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="inferno", ax=ax)

    fig.savefig(out_name, bbox_inches="tight")


def main_norm():
    """
    The main function to normalize the features.
    """
    # Load the features and their names
    data = np.load("./combined_feature_406.npz")
    features = data["features"]
    names = data["names"]
    sids = data["sids"]

    # Generate the mean/std statistics first
    feature_dict, dmso_feature_dict = group_features(features, names)
    compute_stats(feature_dict, dmso_feature_dict, names)

    # Get the mean and sd dictionaries
    dicts = np.load("./z_score_norm_dicts.npz")
    dmso_mean_dict = dicts["dmso_mean_dict"].item()
    dmso_sd_dict = dicts["dmso_sd_dict"].item()

    normed_features = []

    for i in range(len(names)):
        cur_feature = features[i, :]
        cur_name = names[i]
        normed_features.append(z_score_norm(cur_feature,
                                            cur_name,
                                            dmso_mean_dict,
                                            dmso_sd_dict))

    features = None
    normed_features = np.vstack(normed_features)
    np.savez("normed_features.npz", features=normed_features, names=names,
             sids=sids)


def main_vis():
    """
    The main function to visualize batch effects.
    """

    """
    # Load the features and their names before normalization
    data = np.load("./resource/combined_feature_406.npz")
    features = data["features"]
    names = data["names"]

    # Plot the correlation matrix
    feature_dict, dmso_feature_dict = group_features(features, names)
    vis_batch_effect(dmso_feature_dict, out_name="before_correlation.png")

    # Load the features and their names after normalization
    """

    data = np.load("./resource/normed_features.npz")
    features = data["features"]
    names = data["names"]

    # Plot the correlation matrix
    feature_dict, dmso_feature_dict = group_features(features, names)
    vis_batch_effect(dmso_feature_dict, out_name="after_correlation.png")


main_vis()





