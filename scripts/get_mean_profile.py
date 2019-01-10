import tarfile
from os import remove
from shutil import copyfile, rmtree
from os.path import join, exists
from json import load

data_path = "./resource/plate_num.json"
plate_num = load(open(data_path, 'r'))

for pid in plate_num:
    name = "./Plate_{}.tar.gz".format(pid)

    # Copy the tar ball
    copyfile("/mnt/gluster/zwang688/Plate_{}.tar.gz".format(pid), name)

    # Extract the zip file and remove it
    try:
        tar = tarfile.open(name, "r:gz")
        tar.extractall()
        tar.close()
        remove(name)
    except:
        print("Error in {}".format(pid))

        # Clean the dirs
        remove(name)
        if exists("gigascience_upload"):
            rmtree("gigascience_upload")

        continue

    # Copy out the mean profile csv
    copyfile(join("gigascience_upload", "Plate_{}".format(pid), "profiles",
                  "mean_well_profiles.csv"),
             join("tables", "mean_well_profiles_{}.csv".format(pid)))

    # Remove all other files
    rmtree("gigascience_upload")
