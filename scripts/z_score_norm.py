import numpy as np
import re


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


# Get the mean and sd dictionaries
dicts = np.load("./z_score_norm_dicts.npz")
dmso_mean_dict = dicts["dmso_mean_dict"].item()
dmso_sd_dict = dicts["dmso_sd_dict"].item()

# Load the features and their names
#data = np.load("./combined_feature_349_e1.npz")
data = np.load("./test.npz")
features = data["features"]
names = data["names"]
sids = data["sids"]

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

