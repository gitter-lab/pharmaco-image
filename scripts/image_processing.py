"""
A collection of utility functions for image processing.
"""

import sqlite3
import cv2
import numpy as np
from json import dump
from os.path import basename, join


def get_second_key(c, pid, sid, wid):
    """
    Get the second key to query Cells table (TableNumber and ImageNumber).
    Args:
        c: sqlite3 cursor
        pid: int, plate index
        sid: int, site index
        wid: string, well index
    """
    # Get the second combined key
    c.execute(
        """
        SELECT TableNumber, ImageNumber
        FROM Image
        WHERE Image_Metadata_Plate = {} AND Image_Metadata_Site = '{}'
            AND Image_Metadata_Well = '{}'
        """.format(pid, sid, wid)
    )

    result = c.fetchall()
    return result[0]


def get_all_location(c, pid, sid, wid, output=None):
    """
    Get the location data in all cells in the well.
    args:
        c: sqlite3 cursor
        pid: int, plate index
        sid: int, site index
        wid: string, well index
        output: string, the output json path
    return:
        A dict mapping cell index to its location data.
    """
    # Get the total number of cells in this well
    c.execute(
        """
        SELECT Image_Count_Cells
        FROM Image
        WHERE Image_Metadata_Plate = {} AND Image_Metadata_Site = '{}'
                AND Image_Metadata_Well = '{}'
        """.format(pid, sid, wid)
    )

    cell_count = int(c.fetchall()[0][0])
    print(cell_count)

    # Get the secondary key
    tid, iid = get_second_key(c, pid, sid, wid)

    # Get the location data of all cells
    c.execute(
        """
        SELECT Cells_AreaShape_Center_X, Cells_AreaShape_Center_Y,
            Cells_AreaShape_MajorAxisLength, Cells_AreaShape_MinorAxisLength,
            Cells_AreaShape_Orientation
        FROM Cells
        WHERE TableNumber = '{}' AND ImageNumber = {} AND
            ObjectNumber BETWEEN 1 AND {};
        """.format(tid, iid, cell_count)
    )

    location = c.fetchall()

    # Return a dict
    dic = dict(zip(range(1, cell_count + 1), location))

    # Save the dictionary if needed
    if output:
        dump(dic, open(output, 'w'), indent=4, sort_keys=True)

    return dic


def crop_image_from_well(img_name, location, save_dir):
    """
    Crop single cell images from the img_name image.
    args:
        img_name: string, the path to the image
        location: a location dictionary encoding the cell index and position
        save_path: string, the directory name where images are saved
    """

    # We want to preserve the orientation of cells in the combined image, so
    # we need to use a polygon mask there.
    for k in [31]:
        image = cv2.imread(img_name, -1)

        # 1. Make polygon
        pos = location[k]
        x, y = int(pos[0]), int(pos[1])
        half_major, half_minor = round(pos[2] / 2), round(pos[3] / 2)
        degree = pos[4]

        points = np.array([[
            [x - half_major, y - half_minor],
            [x + half_major, y - half_minor],
            [x + half_major, y + half_minor],
            [x - half_major, y + half_minor]
        ]])

        # Rotate all the vertxes around the center point
        rotate_matrix = cv2.getRotationMatrix2D((x, y), -degree, 1)
        rotated_points = cv2.transform(points, rotate_matrix)[0]

        # Format the polygon
        points = rotated_points.astype(int).reshape((-1, 1, 2))

        # 2. Crop the bounding box of the polygon
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()

        # 3. Generate a mask, target is white and others are black
        # The mask image must be in 8-bit format even if we are dealing
        # with 16-bit images
        mask = np.zeros(cropped.shape, dtype=np.uint8)

        # Support colored or gray scale
        channel_count = 1 if len(image.shape) == 2 else image.shape[2]
        white_color = (255,) * channel_count

        # Translate the points into the bounding box axis
        translated_points = points - points.min(axis=0)
        cv2.fillConvexPoly(mask, translated_points, white_color)

        # 4. Apply the mask
        masked_image = cv2.bitwise_and(cropped, cropped, mask=mask)
        #cv2.imshow('tif', cropped * 16)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Save the masked_image to the output directory
        new_name = 'c{:03}_{}'.format(int(k), basename(img_name))
        new_name = join(save_dir, new_name)

        cv2.imwrite(new_name, masked_image * 16)


if __name__ == '__main__':
    sql_path = './data/test/meta_data/extracted_features/24278.sqlite'
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    dic = get_all_location(c, 24278, 1, 'a01', 'test.json')
    crop_image_from_well('./test.tif', dic, './test')
