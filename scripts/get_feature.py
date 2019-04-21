import sklearn
import numpy as np
import pandas as pd
from json import load, dump

# Load the output matrix
output_matrix_npz = np.load("./resource/output_matrix_convert_collision.npz")
compound_inchi = output_matrix_npz["compound_inchi"]
compound_broad_id = output_matrix_npz["compound_broad_id"]
output_matrix = output_matrix_npz["output_matrix"]
output_matrix.shape

# It is slow to work with DataFrame, we can make a dictionary with BID -> PID+WID
mean_well_df = pd.read_csv("./resource/merged_mean_table.csv")

all_pids = mean_well_df["Metadata_Plate"].tolist()
all_wids = mean_well_df["Metadata_Well"].tolist()
all_bids = mean_well_df["Metadata_broad_sample"].tolist()

mean_well_dict = {}
intersect_bids = set(compound_broad_id)
for i in range(len(all_bids)):
    cur_bid = all_bids[i]
    if cur_bid in intersect_bids:
        if cur_bid not in mean_well_dict:
            mean_well_dict[cur_bid] = {"pid": [all_pids[i]],
                                       "wid": [all_wids[i]]}
        else:
            mean_well_dict[cur_bid]["pid"].append(all_pids[i])
            mean_well_dict[cur_bid]["wid"].append(all_wids[i])
     
print("From our 244 plates, we found {} compounds over {} total intersected compounds.".format(
    len(mean_well_dict), len(compound_broad_id)))

# Reorder pids and wids to match output matrix row order
ordered_pids, ordered_wids = [], []
compound_replicate = {}

for b in compound_broad_id:
    if b in mean_well_dict:
        ordered_pids.extend(mean_well_dict[b]["pid"])
        ordered_wids.extend(mean_well_dict[b]["wid"])
        compound_replicate[b] = len(mean_well_dict[b]["pid"])

# Extract features for those plates & wells
pwid = ["{}_{}".format(ordered_pids[i], ordered_wids[i]) for i in range(len(ordered_wids))]
pwid_set = set(pwid)

combined_feature = np.load("/Users/JayWong/Downloads/combined_feature.npz")
combined_feature_pwids = combined_feature["names"]

# Get matching indices
matched_indices = []
matched_pwids = []
for i in range(len(combined_feature_pwids)):
    if combined_feature_pwids[i] in pwid_set:
        matched_indices.append(i)
        matched_pwids.append(combined_feature_pwids[i])

all_feature = combined_feature["features"]
matched_feature = all_feature[matched_indices,:]
matched_names = combined_feature_pwids[matched_indices]
np.savez("./resource/matched_collision_raw_features.npz",
         features=matched_feature, names=matched_names)
