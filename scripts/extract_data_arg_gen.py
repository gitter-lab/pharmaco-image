import numpy as np
from json import dump
from sys import argv


def select_wells(assay):
    """
    Given an assay id, select pid and wid to extract images from.

    Args:
        assay(int): assay index (column of the output matrix)

    Return:
        a dictionary: {pid: [(wid, label), (wid, label)...]}
    """

    # Load the output matrix and each row's corresponding pid, wid
    output_data = np.load('./output_matrix_convert_collision.npz',
                          allow_pickle=True)
    output_matrix = output_data['output_matrix']
    pid_wids = output_data['pid_wids']

    # Find selected compounds in this assay
    selected_index = output_matrix[:, assay] != -1
    selected_index = [i for i in range(len(selected_index)) if
                      selected_index[i]]
    selected_labels = output_matrix[:, assay][selected_index]
    selected_pid_wids = np.array(pid_wids)[selected_index]

    # Flatten the selected pid_wids and group them by pid
    # selected_wells has structure [(wid, pid, label, cpd_index)]
    selected_wells = []

    for i in range(len(selected_pid_wids)):
        cur_pid_wids = selected_pid_wids[i]
        cur_label = selected_labels[i]
        cur_index = selected_index[i]

        for pid_wid in cur_pid_wids:
            selected_wells.append((pid_wid[0], pid_wid[1], int(cur_label),
                                   cur_index))

    # Group these wells by their pids
    selected_well_dict = {}
    for well in selected_wells:
        cur_pid, cur_wid, cur_label, cur_index = (well[0], well[1], well[2],
                                                  well[3])

        if cur_pid in selected_well_dict:
            selected_well_dict[cur_pid].append((cur_wid, cur_label, cur_index))
        else:
            selected_well_dict[cur_pid] = [(cur_wid, cur_label, cur_index)]

    return selected_well_dict


def arg_gen(assay):
    """
    Generate argument file to submit condor jobs. The argument file enables
    multi-argument submission where each node processes one plate.

    For each line, arguments are: assay, pid, zip_1, zip_2, zip_3, zip_4, zip_5

    Args:
        assay(int): assay index
    """

    # Extract associated plate ids for the given assay
    selected_well_dict = select_wells(assay)
    dump(selected_well_dict, open('./selected_well_dict.json', 'w'), indent=2)

    # Write the argument file
    with open('args.txt', 'w') as fp:
        for pid in selected_well_dict:
            fp.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
                assay,
                pid,
                '/mnt/gluster/zwang688/{}-ERSyto.zip'.format(pid),
                '/mnt/gluster/zwang688/{}-ERSytoBleed.zip'.format(pid),
                '/mnt/gluster/zwang688/{}-Hoechst.zip'.format(pid),
                '/mnt/gluster/zwang688/{}-Mito.zip'.format(pid),
                '/mnt/gluster/zwang688/{}-Ph_golgi.zip'.format(pid),
            ))


if __name__ == '__main__':
    assay = int(argv[1])
    arg_gen(assay)
