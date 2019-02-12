from zipfile import ZipFile, BadZipFile
from os.path import join, basename
from glob import glob


def check_all_zip(directory="./", output_f="./corrupted_zip_list.txt"):
    """
    There is no `zip` or `unzip` installed at transfer2000, so one can use
    this script to check the integrity of all zip files in one directory.
    """
    corrupted_list = []
    for name in glob(join(directory, "*.zip")):
        try:
            zf = ZipFile(name, 'r')
            if zf.testzip() is not None:
                print("Not none: {}".format(basename(name)))
                corrupted_list.append("{}\n".format(basename(name)))
        except:
            print(basename(name))
            corrupted_list.append("{}\n".format(basename(name)))

    with open(output_f, 'w') as fp:
        for line in corrupted_list:
            fp.write(line)


check_all_zip()
