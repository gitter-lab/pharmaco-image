"""
A collection of utility functions for image processing.
"""

import sqlite3
import cv2
import random
import string
import os
import re
import numpy as np
from shutil import copyfile, rmtree
from json import dump
from os.path import basename, join, exists
from glob import glob


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


def get_intersect_point(v1, v2, width, height):
    """
    To deal with the out-of-frame polygons, we need to compute
    the intersection points of missing vertex and contained
    vertex to generate a new polygon to crop.

    This is a helper function to compute the intersect point
    of the line between two vertecies and all four edge lines.

    Args:
        v1, v2: each of the coordinate in list form [x1, y1], etc.
        width, height: ints, the width and height of the image
    Rturn:
        [x, y] if found a intersection point, otherwise None
    """

    # Compute the v1v2 line
    x1, y1, x2, y2 = *v1, *v2
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    x_max, x_min = max(x1, x2), min(x1, x2)
    y_max, y_min = max(y1, y2), min(y1, y2)

    # Test the left vertical edge
    if b >= 0 and b <= y_max and b >= y_min:
        return [0, round(b)]

    # Test the right vertical edge
    y_right = width * k + b
    if y_right >= 0 and y_right <= y_max and y_right >= y_min:
        return [width, round(y_right)]

    # Test the top horizontal edge
    x_top = (-1) * b / k
    if x_top >= 0 and x_top <= x_max and x_top >= x_min:
        return [round(x_top), 0]

    # Test the bottom horizontal edge
    x_bot = (height - b) / k
    if x_bot >= 0 and x_bot <= x_max and x_bot >= x_min:
        return [round(x_bot), height]

    return None


def get_new_polygon(original_polygon, width, height):
    """
    Find the intersection polygon of the original polygon and the
    image.
    Args:
        original_polygon: [4,1,2] shape numpy array. The vertex order
                          The order of vertices must be top-left, top-right,
                          bot-right, bot-left.
        width, height: int, the width of height of the image
    Return:
        The original_polygon if the area is fully contained in the
        image. Otherwise a smaller sub-polygon intersected with the
        image.
    """
    new_vertices = []

    for i in range(4):
        vertex = original_polygon[i][0]

        # Test if the vertex is out of range
        if vertex[0] < 0 or vertex[0] > width or vertex[1] < 0 or \
           vertex[1] > height:
            previous_vertex = original_polygon[i - 1][0]
            next_vertex = original_polygon[(i + 1) % 4][0]

            pre_intersect = get_intersect_point(vertex, previous_vertex,
                                                width, height)
            next_intersect = get_intersect_point(vertex, next_vertex,
                                                 width, height)

            if not pre_intersect and not next_intersect:
                # This missing vertex has two missing neighbors
                # We replae it with the closed vertex of the image
                if vertex[0] < 0:
                    if vertex[1] < 0:
                        new_vertices.append([0, 0])
                    else:
                        new_vertices.append([0, height])
                else:
                    if vertex[1] < 0:
                        new_vertices.append([width, 0])
                    else:
                        new_vertices.append([width, height])

            else:
                # Replace this missing vertex by the intersection
                # point(s) in order
                if pre_intersect:
                    new_vertices.append(pre_intersect)

                if next_intersect:
                    new_vertices.append(next_intersect)
        else:
            # This vertex is contained
            new_vertices.append(vertex)

    return np.array(new_vertices).reshape(-1, 1, 2).astype(int)


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
    for k in location:
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

        # 🙁 We also need to deal some polygons that are out of range
        points = get_new_polygon(points, image.shape[1], image.shape[0])

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
        cv2.fillConvexPoly(mask, translated_points, white_color,
                           lineType=cv2.LINE_AA)

        # 4. Apply the mask
        masked_image = cv2.bitwise_and(cropped, cropped, mask=mask)

        # Save the masked_image to the output directory
        new_name = 'c{:03}_{}'.format(int(k), basename(img_name))
        new_name = join(save_dir, new_name)

        cv2.imwrite(new_name, masked_image * 16)


def crop_image_from_well_rotate(img_name, location, save_dir, times16=True):
    """
    Crop single cell images from the img_name image. This function will rotate
    the whole image directly so the relative angle of each cell is not
    perversed.

    args:
        img_name: string, the path to the image
        location: a location dictionary encoding the cell index and position
        save_path: string, the directory name where images are saved
        times16: if to multiply the result image by 16
    """
    for k in location:
        image = cv2.imread(img_name, -1)

        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)

        # Get the bounding box information
        pos = location[k]
        x, y = int(pos[0]), int(pos[1])
        half_major, half_minor = round(pos[2] / 2), round(pos[3] / 2)
        degree = pos[4]

        # Rotate the whole image
        rotate_matrix = cv2.getRotationMatrix2D(image_center, degree, 1)

        # Use cos and sin to compute the smallest rec to contain the rotated
        # image so we are not losing pixels
        abs_cos = abs(rotate_matrix[0, 0])
        abs_sin = abs(rotate_matrix[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Add translation to the rotation matrix
        rotate_matrix[0, 2] += bound_w / 2 - image_center[0]
        rotate_matrix[1, 2] += bound_h / 2 - image_center[1]

        # Transform the whole image
        rotated_image = cv2.warpAffine(image, rotate_matrix,
                                       (bound_w, bound_h))

        # Transform the center
        rotated_x, rotated_y = cv2.transform(np.array([[[x, y]]]),
                                             rotate_matrix)[0][0].astype(int)

        # Get the bounding rectangle
        points = np.array([[
            [rotated_x - half_major, rotated_y - half_minor],
            [rotated_x + half_major, rotated_y - half_minor],
            [rotated_x + half_major, rotated_y + half_minor],
            [rotated_x - half_major, rotated_y + half_minor]
        ]])

        points = points.astype(int).reshape((-1, 1, 2))

        # Crop the rectangle
        x, y, w, h = (rotated_x - half_major, rotated_y - half_minor,
                      half_major * 2, half_minor * 2)

        # Fix out-of-frame problem
        if x >= bound_w or y >= bound_h:
            print("Left point of the cropping rec out of range.")
            exit(1)
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0

        cropped = rotated_image[y: y + h, x: x + w].copy()
        cropped = cropped * 16 if times16 else cropped

        # Save the masked_image to the output directory
        new_name = re.sub(r'^(.*)\.tif$', r'\1' + '_c{:03}.tif'.format(int(k)),
                          basename(img_name))
        new_name = join(save_dir, new_name)

        cv2.imwrite(new_name, cropped)


def make_gray_training_dir(pid, wid, input_dir, output_dir, sql_path):
    """
    Generate a training directory having a subdirectory for each pid+wid. All
    channels and DOF cropped single cell images are stored in those
    subdirectories.

    Args:
        input_dir: the directory storing the 5 channel directories. Each
                   directory is expected to have name like `24278-ERSyto`.
        output_dir: the directory to save cropped images.
    """

    # Create output dir
    out_sub_dir = join(output_dir, '{}_{}'.format(pid, wid))
    if not exists(out_sub_dir):
        os.mkdir(out_sub_dir)

    # Copy the images into a temporary cache dir and give them a better name
    cache_dir = ''.join(random.choice(string.ascii_uppercase)
                        for _ in range(10))
    os.mkdir(cache_dir)
    dies = {'ERSyto': 1,
            'ERSytoBleed': 2,
            'Hoechst': 3,
            'Mito': 4,
            'Ph_golgi': 5}

    cache_images = []
    for d in dies:
        channel_dir = join(input_dir, '{}-{}'.format(pid, d))
        # Get all the DOF images
        images = [f for f in os.listdir(channel_dir) if
                  re.search(r'^.*_{}_s\d_.*\.tif$'.format(wid), f)]
        for i in images:
            dest_path = join(cache_dir,
                             re.sub(r'^.+_(a\d+_s\d).*$',
                                    r'ch{}_\1.tif'.format(dies[d]), i))
            cache_images.append(dest_path)
            src_path = join(channel_dir, i)
            copyfile(src_path, dest_path)

    # Crop images
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()
    for i in cache_images:
        # Get the sid
        sid = int(re.sub(r'^.*ch\d_{}_s(\d).tif$'.format(wid), r'\1', i))
        location = get_all_location(c, pid, sid, wid)
        crop_image_from_well(i, location, out_sub_dir)

    # Clean the cache
    rmtree(cache_dir)


def make_rgb_training_dir(pid, wid, input_dir, output_dir, sql_path):
    """
    Generate a training directory having a subdirectory for each pid+wid. All
    channels and DOF cropped single cell images are stored in those
    subdirectories. Cells are merged into RGB scale by 123 channels, and 45
    channels.

    Args:
        input_dir: the directory storing the 5 channel directories. Each
                   directory is expected to have name like `24278-ERSyto`.
        output_dir: the directory to save cropped images.
    """

    # Create output dir
    if not exists(output_dir):
        os.mkdir(output_dir)
    out_sub_dir = join(output_dir, '{}_{}'.format(pid, wid))
    if not exists(out_sub_dir):
        os.mkdir(out_sub_dir)
        os.mkdir(join(out_sub_dir, 'c123'))
        os.mkdir(join(out_sub_dir, 'c45'))

    # Copy the images into a temporary cache dir and give them a better name
    cache_dir = ''.join(random.choice(string.ascii_uppercase)
                        for _ in range(10))
    os.mkdir(cache_dir)
    os.mkdir(join(cache_dir, 'c123'))
    os.mkdir(join(cache_dir, 'c45'))

    dies = {'ERSyto': 1,
            'ERSytoBleed': 2,
            'Hoechst': 3,
            'Mito': 4,
            'Ph_golgi': 5}

    for sid in range(1, 10):
        channels = []
        for d in dies:
            channel_dir = join(input_dir, '{}-{}'.format(pid, d))
            images = [f for f in os.listdir(channel_dir) if
                      re.search(r'^.*_{}_s{}_.*\.tif$'.format(wid, sid), f)]
            if len(images) == 0:
                # The pid has less than 9 DOF
                break
            channels.append(cv2.imread(join(channel_dir, images[0]), -1) * 16)

        # Merge the channels and save to the cache
        if len(channels) == 0:
            break
        black_image = np.zeros(channels[0].shape).astype(channels[0].dtype)
        c123_name = join(cache_dir, 'c123/{}_{}_{}.tif'.format(pid, wid, sid))
        c45_name = join(cache_dir, 'c45/{}_{}_{}.tif'.format(pid, wid, sid))
        cv2.imwrite(c45_name, cv2.merge([channels[4],
                                         black_image,
                                         channels[3]]))
        cv2.imwrite(c123_name, cv2.merge([channels[2],
                                          channels[1],
                                          channels[0]]))

    # Crop images
    conn = sqlite3.connect(sql_path)
    c = conn.cursor()

    # Crop c123
    for i in glob(join(cache_dir, 'c123/*.tif')):
        sid = int(re.sub(r'^.*_.*_(\d).tif$'.format(wid), r'\1', i))
        location = get_all_location(c, pid, sid, wid)
        crop_image_from_well_rotate(i, location, join(out_sub_dir, 'c123'),
                                    times16=False)

    # Crop c45
    for i in glob(join(cache_dir, 'c45/*.tif')):
        sid = int(re.sub(r'^.*_.*_(\d).tif$'.format(wid), r'\1', i))
        location = get_all_location(c, pid, sid, wid)
        crop_image_from_well_rotate(i, location, join(out_sub_dir, 'c45'),
                                    times16=False)

    rmtree(cache_dir)


if __name__ == '__main__':
    sql_path = './data/test/meta_data/extracted_features/24278.sqlite'

    make_rgb_training_dir(24278, 'a15', '/Users/JayWong/Downloads', './train',
                          sql_path)
    make_rgb_training_dir(24278, 'j12', '/Users/JayWong/Downloads', './train',
                          sql_path)
    make_rgb_training_dir(24278, 'p22', '/Users/JayWong/Downloads', './train',
                          sql_path)
    make_rgb_training_dir(24278, 'a13', '/Users/JayWong/Downloads', './train',
                          sql_path)
