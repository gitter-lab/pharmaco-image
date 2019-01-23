import re
from math import ceil


def distribute_plates(plate_list, window_size, output_f):
    """
    Distribute plates into subsamples, each subsample has window_size number
    of jobs. The results will be written to output_f with the plate numbers
    organized in one line for each sample.
    """

    # Extract the pids
    with open(plate_list, 'r') as fp:
        lines = [line.replace('\n', '') for line in fp.readlines()]
        pids = []
        for line in lines:
            pids.append(int(re.sub(r'Plate_(\d+)\.tar\.gz', r'\1', line)))

    # Partition the pids
    outputs = ""
    for i in range(ceil(len(pids) / window_size)):
        cur_line = ' '.join(map(
            str,
            pids[i * window_size: min((i + 1) * window_size, len(pids))]
        ))
        outputs += cur_line + '\n'

    with open(output_f, 'w') as fp:
        fp.write(outputs)


distribute_plates("all_plates_406.txt", 40, "args.txt")
