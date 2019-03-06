#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:14:39 2018

@author: mika

This file contains an example implementation of how the images can be read
in python.

I give no guarantees of the implementation working properly.
"""
from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import imageio
from orderedset import OrderedSet

def get_croppedyale_as_df():
    """ Get a subset of the cropped Yale images as one 2D array (Pandas DataFrame).

    All 'Ambient' images are discarded. Also, if there are persons that have not
    been photographed in certain illumination conditions, none of the images taken
    in those conditions are included in the result.

    Each row of the output table (Pandas data frame) contains one image represented
    as a linear array.
    """
    pics, all_suffixes, suffixes = load_images_croppedyale()
    feature_matrix, person_names, full_names, resolution = images_to_array(pics, suffixes)
    name_suffixes = [full_name[1] for full_name in full_names]
    feature_matrix = pd.DataFrame(feature_matrix)
    feature_matrix.loc[:, 'person'] = person_names
    feature_matrix.loc[:, 'pic_name'] = name_suffixes
    feature_matrix = feature_matrix.set_index(['person', 'pic_name'])
    return feature_matrix, resolution

def load_images_croppedyale():
    """ Read images from the cropped Yale data set.

    The implementation is heavily tied to the known directory structure of the data set.
    Also, pictures labeled 'Ambient' are discarded because some of them have not been
    cropped and are not the same size as the other images.

    Returns:
    1) The loaded pictures as an ordered dictionary of ordered dictionaries.
    The keys of the first-level dictionary are folder names such as 'yaleB10', each name
    corresponding to one person. In the second-level dictionaries the keys are strings
    describing the pose and the illumination, as extracted from the file names.
    2) All keys appearing in any of the second-level dictionaries, as an OrderedSet.
    3) All keys appearing in all second-level dictionaries, as an OrderedSet.
    """
    orig_folder = os.getcwd()
    try:
        os.chdir('CroppedYale')
        subdirs = sorted(os.listdir())
        pics = OrderedDict()
        u_fname_suffixes, i_fname_suffixes = OrderedSet(), OrderedSet()
        image_resolution = None
        first_iter = True
        for subdir in subdirs: # Iterate over all folders (persons)
            pics[subdir] = OrderedDict()
            os.chdir(subdir)
            file_names = [x for x in os.listdir() if x.endswith('.pgm') and 'Ambient' not in x]
            current_suffixes = sorted(x[len(subdir) + 1 : -4] for x in file_names)
            for fname_suffix in current_suffixes: # Iterate over files (images)
                fname = subdir + '_' + fname_suffix + '.pgm'
                pic = imageio.imread(fname)
                pics[subdir][fname_suffix] = pic
                if image_resolution is None:
                    image_resolution = [len(pic[0]), len(pic)]
                elif image_resolution != [len(pic[0]), len(pic)]:
                    print('Warning: input images have different sizes.')
            u_fname_suffixes = u_fname_suffixes.union(current_suffixes)
            if first_iter:
                i_fname_suffixes, first_iter = u_fname_suffixes, False
            else:
                i_fname_suffixes = i_fname_suffixes.intersection(current_suffixes)
            os.chdir('..')
        return pics, u_fname_suffixes, i_fname_suffixes
    finally:
        os.chdir(orig_folder)

def images_to_array(pics, included_suffixes):
    """ Convert the given pictures to a numpy array.

    Each row of the returned array is a linear representation of one image.
    Pictures whose properties do not match any of included_suffixes are not included.
    It is also checked that all images have the same resolution - if not, an
    Exception is raised.

    In addition to the array of pictures, also a list of all labels (person
    'names') and suffixes (lighting conditions) is returned, as well as the
    resolution of the images.
    """
    x_res, y_res = -1, -1
    feature_matrix = None
    person_names = []
    full_pic_names = []
    # Get the resolution and the names of the included pictures. For
    # better performance we defer the creation of the feature matrix.
    for person_name in pics.keys():
        for pic_name in included_suffixes:
            current_pic = pics[person_name][pic_name]
            xpix, ypix = len(current_pic[0]), len(current_pic)
            if x_res < 0: # We are processing the first image
                x_res, y_res = xpix, ypix
            else:
                if (x_res, y_res) != (xpix, ypix):
                    raise Exception('All images must be of the same size.')
            person_names += [person_name]
            full_pic_names += [(person_name, pic_name)]
    # Get the pictures into the feature matrix. Pixel intensities are
    # scaled from 0...255 to the range [0, 1].
    pic_arrays = [np.array((pics[s[0]][s[1]] / 255).ravel()) for s in full_pic_names]
    feature_matrix = np.vstack(pic_arrays)
    return feature_matrix, np.array(person_names), np.array(full_pic_names), [x_res, y_res]
