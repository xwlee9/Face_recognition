#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:39:40 2018

@author: mika

This is a simple example where we read the images as a DataFrame using
get_croppedyale_as_df() and then select some rows of the DataFrame based
on person 'names' or lighting conditions.
"""

import read_yale

images, resolution = read_yale.get_croppedyale_as_df()
# The data frame uses a MultiIndex to store the person 'names' and lighting
# conditions. Here we briefly demonstrate using the data frame.

# Get the names of the persons
row_persons = images.index.get_level_values('person')
# Get all images of 'yaleB10'
rows_include = (row_persons == 'yaleB10')
pics_B10 = images[rows_include]
print(pics_B10) # there are over 30 000 columns so results are not pretty..
# Get all images under conditions "P00A-130E+20"
row_conds = images.index.get_level_values('pic_name')
rows_include = (row_conds == 'P00A-130E+20')
pics_2 = images[rows_include]
print(pics_2)
