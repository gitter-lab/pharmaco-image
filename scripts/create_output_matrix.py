import numpy as np
import pandas as pd
from collections import Counter

# Load the output matrix
output_matrix_npz = np.load("./output_matrix_convert_collision.npz")
compound_inchi = output_matrix_npz["compound_inchi"]
compound_broad_id = output_matrix_npz["compound_broad_id"]
output_matrix = output_matrix_npz["output_matrix"]
output_matrix.shape

# It is slow to work with DataFrame, we can make a dictionary with BID -> PID+WID
mean_well_df = pd.read_csv("./merged_table_406.csv")

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
     
print("From all 406 plates, we found {} compounds over {} total intersected compounds.".format(
    len(mean_well_dict), len(compound_broad_id)))

mean_weill_df = None

# Reorder pids and wids to match output matrix row order
ordered_pids, ordered_wids = [], []
compound_replicate = {}

for b in compound_broad_id:
    if b in mean_well_dict:
        ordered_pids.extend(mean_well_dict[b]["pid"])
        ordered_wids.extend(mean_well_dict[b]["wid"])
        compound_replicate[b] = len(mean_well_dict[b]["pid"])

print(Counter(compound_replicate.values()))

# Extract features for those plates & wells
pwid = ["{}_{}".format(ordered_pids[i], ordered_wids[i]) for i in range(len(ordered_wids))]
pwid_set = set(pwid)

combined_feature = np.load("./normed_features.npz")
combined_feature_pwids = combined_feature["names"]

# Get matching indices
matched_indices = []
matched_pwids = []
for i in range(len(combined_feature_pwids)):
    if combined_feature_pwids[i] in pwid_set:
        matched_indices.append(i)
        matched_pwids.append(combined_feature_pwids[i])
        
print("{} images out of {} images are used".format(len(matched_indices), len(combined_feature_pwids)))

all_feature = combined_feature["features"]
matched_feature = all_feature[matched_indices,:]
matched_names = combined_feature_pwids[matched_indices]

np.savez("./matched_collision_raw_features.npz", features=matched_feature, names=matched_names)

# Rearrange output matrix to match our feature matrix
print(matched_feature.shape)

pwd_to_b_dict = {}
for k, v in mean_well_dict.items():
    for i in range(len(v["pid"])):
        pwd = "{}_{}".format(v["pid"][i], v["wid"][i])
        if pwd in pwd_to_b_dict:
            print(pwd)
        pwd_to_b_dict[pwd] = k
        
output_compound_b_to_index = dict(zip(compound_broad_id, range(len(compound_broad_id))))

# Create the indices to select entries from output matrix
select_bids = []
select_indices = []
for pw in matched_names:
    bid = pwd_to_b_dict[pw]
    select_bids.append(bid)
    select_indices.append(output_compound_b_to_index[bid])

print("{} compounds found in our extracted features.".format(len(set(select_bids))))

# Rearrange output matrix
rearranged_output_matrix = output_matrix[select_indices,:]
rearranged_output_matrix.shape
np.savez("./output_matrix_collision_inception.npz", output_matrix=rearranged_output_matrix)
