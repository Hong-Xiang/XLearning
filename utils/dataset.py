#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-10-28 15:08:21

@author: HongXiang

General dataset manipulations. The minimum operating unit is folder.
   rename all items in a folder
   merge multiple folders
"""

from __future__ import absolute_import, division, print_function
import os
import sys
import re
import random
import argparse
import time
import struct

import scipy.misc
import numpy as np


from six.moves import xrange
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp
import xlearn.utils.image as uti

IMAGE_SUFFIX = ['png', 'jpg']

def rename(folder_name, prefix=None):
    """Rename all files in a folder, changing its prefix, keep its suffix.
    """
    folder_name = os.path.abspath(folder_name)
    if prefix is None:
        pipe_random_prefix = utp.RandomPrefix()
        listing = os.listdir(folder_name)
        for infile in listing:
            prefix = next(pipe_random_prefix.out)
            if os.path.isdir(infile):
                continue
            fname_new = prefix + infile
            old_name = os.path.join(folder_name, infile)
            new_name = os.path.join(folder_name, fname_new)
            os.rename(old_name, new_name)
    else:
        print('renaming:', folder_name)
        rename(folder_name)
        listing = os.listdir(folder_name)
        cid = 0
        for infile in listing:
            if os.path.isdir(infile):
                continue
            index = infile.rfind('.')
            if index < len(infile) - 1:
                suffix = infile[index + 1:]

            else:
                suffix = ''
            fname_new = utg.form_file_name(prefix, cid, suffix)
            old_name = os.path.join(folder_name, infile)
            new_name = os.path.join(folder_name, fname_new)
            os.rename(old_name, new_name)
            cid += 1


def img2jpeg(folder_name):
    path = os.path.abspath(folder_name)
    files = os.listdir(path)
    for file in files:
        prefix, id, suffix = utg.seperate_file_name(file)
        if suffix == 'jpg':
            continue
        fullname = os.path.join(path, file)
        # im = Image.open(fullname)
        im = scipy.misc.imread(fullname)
        if im.mode != 'RGB':
            im.convert('RGB')
        newname = utg.form_file_name(prefix, id, 'jpg')
        print('convert:', file, newname)

        fullnamenew = os.path.join(path, newname)
        im.save(fullnamenew, 'JPEG')


def jpg2npy(folder_name, prefix, id0, id1):
    print("JPEG to NPY Tool...")
    path = os.path.abspath(folder_name)
    files = os.listdir(path)
    ids = list(xrange(int(id0), int(id1) + 1))

    for file in files:
        fullname = os.path.join(path, file)
        if os.path.isdir(fullname):
            continue
        prefix_f, id_f, suffix_f = utg.seperate_file_name(file)

        if prefix_f != prefix:
            continue
        if suffix_f != 'jpg':
            continue
        if id_f not in ids:
            continue
        im = scipy.misc.imread(fullname)
        im = np.array(im)
        filename = utg.form_file_name(prefix, id_f, 'npy')
        np.save(filename, im)

def load_raw(filename, shape):
    """load raw data of single type."""
    height = shape[0]
    width = shape[1]
    frame = shape[2]
    pixel = height * width * frame
    data = np.zeros([pixel])
    with open(filename) as f:
        bindata = f.read()
        for i in xrange(pixel):
            res = struct.unpack('<f', bindata[i * 4:i * 4 + 4])
            data[i] = res[0]
    data2 = np.zeros([height, width, frame])
    for i in range(width):
        for j in range(height):
            for k in range(frame):
                idf = i + j * width + k * width * height
                data2[j, i, k] = data[idf]
    return data2

def raw2npy(folder_name, shape, prefix, **kwargs):
    """convert all raw files into npy files.
    """
    path = os.path.abspath(folder_name)
    files = os.listdir(path)

    if 'suffix' in kwargs:
        suffix = kwargs['suffix']
    else:
        suffix = ''
    if 'id0' in kwargs:
        id0 = kwargs['id0']
    else:
        id0 = 0
    if 'id1' in kwargs:
        id1 = kwargs['id1']
    else:
        id1 = len(files)

    ids = list(xrange(int(id0), int(id1) + 1))
    for file in files:
        fullname = os.path.join(path, file)
        if os.path.isdir(fullname):
            continue
        prefix_f, id_f, suffix_f = utg.seperate_file_name(file)
        if prefix_f != prefix:
            continue
        if suffix_f != suffix:
            continue
        if id_f not in ids:
            continue
        height = shape[0]
        width = shape[1]
        frame = shape[2]
        pixel = height * width * frame
        data = np.zeros([pixel])
        with open(fullname) as f:
            bindata = f.read()
            for i in xrange(pixel):
                res = struct.unpack('<f', bindata[i * 4:i * 4 + 4])
                data[i] = res[0]
        data2 = np.zeros([height, width, frame])
        for i in range(width):
            for j in range(height):
                for k in range(frame):
                    idf = i + j * width + k * width * height
                    data2[j, i, k] = data[idf]
        filename = utg.form_file_name(prefix_f, id_f, 'npy')
        np.save(filename, data2)

def proj2sino(folder, prefix_old, prefix_new, id0, id1, folder_out=None):
    if folder_out is None:
        folder_out = folder
    pipe_input = utp.NPYReader(folder, prefix_old, ids=xrange(id0, id1))
    pipe_sino = utp.Proj2Sino(pipe_input)
    pipe_count = utp.Counter()
    pipe_output = utp.NPYWriter(folder_out, prefix_new, pipe_sino, pipe_count)


def sion2proj(folder, prefix_old, prefix_new, id0, id1, folder_out=None):
    if not hasattr(input_, "__iter__"):
        input_ = [input_]
    height = len(input_)
    width, nangles = input_[0].shape
    output = np.zeros([height, width, nangles])
    for iz in xrange(height):
        for iwidth in range(width):
            for iangle in range(nangles):
                output[iz, iwidth, iangle] = input_[iz][iwidth, iangle]
    return output

def main(argv):
    print('Dataset tools for command line.')
    parser = argparse.ArgumentParser(description='Dataset construction.')
    parser.add_argument('--rename',
                        action='store_true',
                        default=False,
                        help='rename all data items.')
    parser.add_argument('--img2jpg',
                        action='store_true',
                        default=False,
                        help='convert all images to JPEG.')
    parser.add_argument('--jpg2npy',
                        action='store_true',
                        default=False,
                        help='convert all JPEG to npy.')
    parser.add_argument('--raw2npy',
                        action='store_true',
                        default=False,
                        help='read raw data to npy')
    parser.add_argument('--filename',
                        action='store_true',
                        default=False,
                        help='Analysis file name.')
    parser.add_argument('--noaction', '-n',
                        action='store_true',
                        default=False,
                        help='print setting, do noting.')

    parser.add_argument('--source', '-s',
                        dest='source',
                        required=True,
                        help='working folder/file.')
    parser.add_argument('--suffix',
                        dest='suffix', default='',
                        help='suffix')
    parser.add_argument('--prefix', dest='prefix')
    parser.add_argument('--shape', dest='shape', nargs='+', type=int)
    parser.add_argument('--index0', dest='id0', type=int)
    parser.add_argument('--index1', dest='id1', type=int)
    parser.add_argument('--endian',
                        dest='endian',
                        default='l',
                        help='Endianness, l or b')

    args = parser.parse_args(argv)
    if args.noaction:
        print(args)
        return
    if args.filename:
        print(utg.seperate_file_name(args.source))
        return

    if args.rename:
        rename(args.source, args.prefix)

    if args.img2jpg:
        img2jpeg(args.source)

    if args.jpg2npy:
        jpg2npy(args.source, args.prefix, args.id0, args.id1)

    if args.raw2npy:
        raw2npy(args.source, args.prefix, args.suffix,
                args.id0, args.id1, args.shape)

    # if argv[1] == 'rename':

    #     rename(argv[2], argv[3])
    # if argv[1] == 'jpeg':
    #     to_jpeg(argv[2])
    # if argv[1] == 'npy':
    #     to_npy(argv[2], argv[3], argv[4], argv[5], argv[6])

if __name__ == "__main__":

    main(sys.argv[1:])
