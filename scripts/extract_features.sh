#!/bin/bash

# Make a temp directory to hold resource files
mkdir data
# A directory to store results
mkdir output

# We extract feature from one plate at one time
for plate in 24277 24278 24279 24280 24293 24294 24295 24296 24297 24300 24301 24302 24303 24304 24305 24306 24307 24308 24309 24310;
do
    for channel in Hoechst ERSyto ERSytoBleed Ph_golgi Mito;
    do
        # Copy and unzip image files from Gluster to the working directory
        cp /mnt/gluster/zwang688/${plate}-${channel}.zip ./data
        unzip -qq ./data/${plate}-${channel}.zip -d ./data
        rm ./data/${plate}-${channel}.zip
    done

    # Copy and unzip the meta data file
    echo 'Working on ${channel} of ${plate}'
    cp /mnt/gluster/zwang688/Plate_${plate}.tar.gz ./data
    tar -xzf ./data/Plate_${plate}.tar.gz -C ./data
    rm ./data/Plate_${plate}.tar.gz

    # Run the python script to extract features
    python extract_features.py ${plate} ./data ./output \
        ./data/gigascience_upload/Plate_${plate}/extracted_features/${plate}.sqlite \
        ./data/gigascience_upload/Plate_${plate}/profiles/mean_well_profiles.csv \
        8

    # Exit the script if python return error
    ret=$?
    if [ $ret -ne 0 ]; then
        exit
    fi

    # Clean the temp data file
    rm -r ./data/*
done

rm -r data
