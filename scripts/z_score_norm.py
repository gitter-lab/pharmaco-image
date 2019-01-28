import numpy as np
import re


def compute_stats(features, names):
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

    # Group the features by pid and DMSO
    dmso_feature_dict = {}
    feature_dict = {}
    for p in pids:
        dmso_feature_dict[p] = []
        feature_dict[p] = []

    dmso_mean_dict = {}
    dmso_sd_dict = {}
    problem_sd_dict = {}

    # Add those features in the dictionary
    for i in range(len(names)):
        pid = int(re.sub(r'(\d{5})_.+', r'\1', names[i]))
        if "DMSO" in names[i]:
            dmso_feature_dict[pid].append(features[i, :])
        feature_dict[pid].append(features[i, :])

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


def main():
    """
    The main function.
    """
    # Load the features and their names
    data = np.load("./combined_feature_406.npz")
    features = data["features"]
    names = data["names"]
    sids = data["sids"]

    # Generate the mean/std statistics first
    compute_stats(features, names)

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


main()
