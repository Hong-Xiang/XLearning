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
import shutil
import scipy.misc
import numpy as np


from six.moves import xrange
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp
import xlearn.utils.image as uti

IMAGE_SUFFIX = ['png', 'jpg']


def rename(source, prefix=None, suffix=None, recurrent=False, no_action=False, force=False):
    """Rename all files in a folder, reform prefixes, with filter using suffix.
    Only files with form: name[.suffix] will be checked.
    If prefix is None, a random prefix will be added.
    """
    folder_name = source
    if isinstance(folder_name, (list, tuple)):
        for folder in folder_name:
            rename(folder, prefix, suffix, recurrent, no_action, force)
        return None
    folder_name = os.path.abspath(folder_name)
    files_maybe = os.listdir(folder_name)

    files_full = list(map(lambda filename: os.path.join(
        folder_name, filename), files_maybe))

    files = list(filter(os.path.isfile, files_full))
    files = list(map(os.path.basename, files))

    if suffix is not None and not force:

        for file_old in files:

            _, _, suffix_old = utg.seperate_file_name(file_old)
            if suffix_old != suffix:
                files.remove(file_old)

    dirs = filter(os.path.isdir, files_full)
    dirs = list(map(os.path.basename, dirs))
    if recurrent:
        for dir_ in dirs:
            fullpath = os.path.join(folder_name, dir_)
            rename(fullpath, prefix, suffix, recurrent, no_action)
    if prefix is None:
        print("RENAME>>>RANDOM PREFIX: {}".format(folder_name))
        pipe_random_prefix = utp.RandomPrefix()
        for file_old in files:
            prefix = next(pipe_random_prefix.out)
            file_new = prefix + file_old
            full_old = os.path.join(folder_name, file_old)
            full_new = os.path.join(folder_name, file_new)
            if no_action:
                print("rename: {0} TO: {1}.".format(full_old, full_new))
            else:
                os.rename(full_old, full_new)
    else:
        print("RENAME>>>FORMAL RENAME: {}".format(folder_name))
        cid = 0
        for file_old in files:
            _, _, suffix_old = utg.seperate_file_name(file_old)
            file_new = utg.form_file_name(prefix, cid, suffix_old)
            full_old = os.path.join(folder_name, file_old)
            full_new = os.path.join(folder_name, file_new)
            if no_action:
                print("rename: {0} TO: {1}.".format(full_old, full_new))
            else:
                os.rename(full_old, full_new)
            cid += 1


def img2jpg(folder_name, prefix, ids=None):
    """Convert all image files under a folder into jpg files."""
    path = os.path.abspath(folder_name)
    files = os.listdir(path)
    for file in files:
        prefix, id, suffix = utg.seperate_file_name(file)
        if suffix == 'jpg':
            continue
        if ids is not None and id not in ids:
            continue
        fullname = os.path.join(path, file)
        img = uti.imread(fullname)
        if img.mode != 'RGB':
            img.convert('RGB')
        newname = utg.form_file_name(prefix, id, 'jpg')
        fullnamenew = os.path.join(path, newname)
        img.save(fullnamenew, 'JPEG')


def jpg2npy(folder_name, prefix, ids=None):
    path = os.path.abspath(folder_name)
    pipe_reader = utp.FolderReader(path, prefix=prefix, ids=ids, suffix='jpg')
    pipe_counter = utp.Counter()
    pipe_writer = utp.FolderWriter(path, prefix, pipe_reader, pipe_counter)
    pipe_runner = utp.Runner(pipe_writer)
    pipe_runner.run()


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


def sino3d2sino2d(source, target, prefix):
    if isinstance(source, (list, tuple)):
        for folder in source:
            sino3d2sino2d(folder, target, prefix)
        return None
    pipe_filename = utp.FileNameLooper(source, prefix)
    pipe_sino3d = utp.NPYReaderSingle(pipe_filename)
    pipe_conv3d2d = utp.Sino3D2Sino2D(pipe_sino3d)
    pipe_buffer = utp.Buffer(pipe_conv3d2d)
    pipe_counter = utp.Counter()
    pipe_sino2dwirter = utp.FolderWriter(
        target, prefix, pipe_buffer, pipe_counter)

    pipe_runner = utp.Runner(pipe_sino2dwirter)
    pipe_runner.run()


def raw2npy(folder_name, shape, prefix, **kwargs):
    """convert all raw files into npy files.
    """
    if isinstance(folder_name, (list, tuple)):
        for folder in folder_name:
            raw2npy(folder, shape, prefix, **kwargs)
        return None
    path = os.path.abspath(folder_name)
    files = os.listdir(path)

    if 'suffix' in kwargs:
        suffix = kwargs['suffix']
    else:
        suffix = ''
    if 'id0' in kwargs:
        id0 = kwargs['id0']
    else:
        id0 = None
    if id0 is None:
        id0 = 0
    if 'id1' in kwargs:
        id1 = kwargs['id1']
    else:
        id1 = None
    if id1 is None:
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
        print("processing: {}.".format(fullname))
        if shape is None:
            raise ValueError('shape is not defined.')

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
        fullname = os.path.join(folder_name, filename)
        np.save(fullname, data2)        


def proj2sino(folder, prefix_old, prefix_new, id0, id1, folder_out=None):
    # TODO: Reimplementation
    if folder_out is None:
        folder_out = folder
    pipe_input = utp.FolderReader(folder, prefix_old, ids=xrange(id0, id1))
    pipe_sino = utp.Proj2Sino(pipe_input)
    pipe_count = utp.Counter()
    pipe_output = utp.FolderWriter(
        folder_out, prefix_new, pipe_sino, pipe_count)


def sion2proj(source, target, prefix_new, id0, id1, folder_out=None):
    # TODO: Reimplementation
    # if not hasattr(input_, "__iter__"):
    #     input_ = [input_]
    # height = len(input_)
    # width, nangles = input_[0].shape
    # output = np.zeros([height, width, nangles])
    # for iz in xrange(height):
    #     for iwidth in range(width):
    #         for iangle in range(nangles):
    #             output[iz, iwidth, iangle] = input_[iz][iwidth, iangle]
    # return output
    pass


def pad_sino(source, target, prefix, pwx, pwy):
    if isinstance(source, (list, tuple)):
        for folder in source:
            pad_sino(folder, target, prefix, pwx, pwy)
        return None
    pipe_filename = utp.FileNameLooper(source, prefix)
    pipe_sino2d = utp.NPYReaderSingle(pipe_filename)
    pipe_padded = utp.PeriodicalPadding(pipe_sino2d, pwx, pwx, pwy, pwy)
    pipe_counter = utp.Counter()
    pipe_sino2dwirter = utp.FolderWriter(
        target, prefix, pipe_padded, pipe_counter)
    pipe_runner = utp.Runner(pipe_sino2dwirter)
    pipe_runner.run()


def combine_infer(source, target, prefix, pwx, pwy):
    if isinstance(source, (list, tuple)):
        source = source[0]
    source = os.path.abspath(source)
    files = os.listdir(source)
    filesfull = []
    for file_ in files:
        fullname = os.path.join(source, file_)
        if os.path.isdir(fullname):
            continue
        prefix_tmp, _, _ = utg.seperate_file_name(file_)
        if prefix_tmp != prefix:
            continue
        filesfull.append(fullname)
    tmp = np.array(np.load(filesfull[0]))
    _, height, width, _ = tmp.shape
    frames = len(filesfull)
    height -= pwx * 2
    width -= pwy * 2
    sino3d = np.zeros([width, height, frames])
    for i in xrange(frames):
        filename = prefix + "%09d.npy" % i
        fullname = os.path.join(source, filename)
        sino2d = np.load(fullname)
        for ix in xrange(height):
            for iy in xrange(width):
                sino3d[iy, ix, i] = sino2d[0, ix + pwx, iy + pwy, 0]
    savename = os.path.join(os.path.abspath(target), 'srresult.npy')
    np.save(savename, sino3d)


def print_std_filename(file):
    if isinstance(file, (list, tuple)):
        map(print_std_filename, file)
    prefix, id_, suffix = utg.seperate_file_name(file)
    print("Analysic filename:{}".format(file))
    print("+P:{0}, I:{1}, S:{2}.".format(prefix, id_, suffix))
    name_std = utg.form_file_name(prefix, id_, suffix)
    print("+STD:{}".format(name_std))


def merge(source, target, random_rename=True, copy_file=False, no_action=False, recurrent=False, prefix=None, suffix=None):
    print("merge tool called on {0}".format(source))
    if random_rename:
        rename(source, prefix=None, suffix=suffix,
               recurrent=recurrent, no_action=no_action)

    if isinstance(source, (list, tuple)):
        map(lambda folder: merge(folder, target, random_rename, copy_file, no_action, recurrent, prefix, suffix),
            source)
        return None

    folder_name = os.path.abspath(source)
    target_full = os.path.abspath(target)
    files_maybe = os.listdir(folder_name)
    files_full = list(map(lambda filename: os.path.join(
        folder_name, filename), files_maybe))

    files = list(filter(os.path.isfile, files_full))
    files = list(map(os.path.basename, files))
    if suffix is not None:
        for file_old in files:
            _, _, suffix_old = utg.seperate_file_name(file_old)
            if suffix_old != suffix:
                files.remove(file_old)

    dirs = list(filter(os.path.isdir, files_full))
    dirs = list(map(os.path.basename, dirs))

    if recurrent:
        for dir_ in dirs:
            fullpath = os.path.join(folder_name, dir_)
            merge(fullpath, target_full, random_rename, copy_file,
                  no_action, recurrent, prefix, suffix)

    for file_old in files:

        full_old = os.path.join(folder_name, file_old)
        full_new = os.path.join(target_full, file_old)
        action = "COPY" if copy_file else "MOVE"
        if no_action:
            print("{0}: {1} TO: {2}.".format(action, full_old, full_new))
        else:
            if copy_file:
                shutil.copyfile(full_old, full_new)
            else:
                os.rename(full_old, full_new)


def main(argv):
    """command line support.
    """

    print('Dataset tools for command line.')
    parser = argparse.ArgumentParser(description='Dataset construction.')
    parser.add_argument('--rename',
                        action='store_true',
                        default=False,
                        help='rename all data items.')
    parser.add_argument('--merge',
                        action='store_true',
                        default=False,
                        help='merge multiple folders.')
    parser.add_argument('--img2jpg',
                        action='store_true',
                        default=False,
                        help='convert all images to JPEG.')
    parser.add_argument('--sino3d2sino2d',
                        action='store_true',
                        default=False,
                        help='Convert 3D sinograms to multiple 2D sinograms.')
    parser.add_argument('--combine_infer',
                        action='store_true',
                        default=False,
                        help='combine inference sino2d to sino3d.')
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
    parser.add_argument('--print',
                        action='store_true',
                        default=False,
                        help='print args, do nothing.')
    parser.add_argument('--pad_sino',
                        action='store_true',
                        default=False,
                        help='padding sinograms.')

    parser.add_argument('--no_action', '-n',
                        action='store_true',
                        default=False,
                        help='print settings, do noting.')
    parser.add_argument('--force', '-f',
                        action='store_true',
                        default=False,
                        help='force doing.')
    parser.add_argument('--recurrent', '-r',
                        action='store_true',
                        default=False,
                        help='recurrent on sub folders.')
    parser.add_argument('--copy_file',
                        action='store_true',
                        default=False,
                        help='copy files instead of move files.')
    parser.add_argument('--random_rename',
                        action='store_true',
                        default=False,
                        help='Add random prefix before merge.')

    parser.add_argument('--source', '-s', dest='source', required=True, nargs='+',
                        help='source folder/file.')
    parser.add_argument('--target', '-t',
                        dest='target', default=None,
                        help='target folder/file.')
    parser.add_argument('--suffix', dest='suffix', default=None, help='suffix')
    parser.add_argument('--prefix', dest='prefix', default=None, help='prefix')
    parser.add_argument('--prefix_new', dest='prefix_new',
                        default=None, help='new prefix')

    parser.add_argument('--shape', dest='shape', nargs='+', type=int)
    parser.add_argument('--index0', dest='id0', type=int)
    parser.add_argument('--index1', dest='id1', type=int)
    parser.add_argument('--endian', dest='endian',
                        default='l', help='Endianness, l or b')

    parser.add_argument('--pwx', type=int, help='padding window x')
    parser.add_argument('--pwy', type=int, help='padding window y')

    args = parser.parse_args(argv)
    if args.print:
        print(args)
        return

    if args.filename:
        for filename in args.source:
            print(utg.seperate_file_name(filename))
        return

    if args.rename:
        rename(args.source, args.prefix, args.suffix,
               args.recurrent, args.no_action, args.force)

    if args.merge:
        merge(args.source, args.target, args.random_rename, args.copy_file,
              args.no_action, args.recurrent, args.prefix, args.suffix)

    if args.pad_sino:
        pad_sino(args.source, args.target, args.prefix, args.pwx, args.pwy)

    if args.combine_infer:
        combine_infer(source=args.source, target=args.target,
                      prefix=args.prefix, pwx=args.pwx, pwy=args.pwy)

    # if args.img2jpg:
    #     img2jpeg(args.source, args.prefix)

    # if args.jpg2npy:
    #     jpg2npy(args.source, args.prefix, args.id0, args.id1)

    if args.raw2npy:
        raw2npy(args.source, shape=args.shape, prefix=args.prefix,
                suffix=args.suffix, id0=args.id0, id1=args.id1)

    if args.sino3d2sino2d:
        sino3d2sino2d(args.source, args.target, args.prefix)

    # if argv[1] == 'rename':

    #     rename(argv[2], argv[3])
    # if argv[1] == 'jpeg':
    #     to_jpeg(argv[2])
    # if argv[1] == 'npy':
    #     to_npy(argv[2], argv[3], argv[4], argv[5], argv[6])

if __name__ == "__main__":

    main(sys.argv[1:])
