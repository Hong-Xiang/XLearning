#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-01 14:32:06
@author: HongXiang

[COMMAND LINE] Merge different data folders into one.
"""
from __future__ import absolute_import, division, print_function
import sys
import os
import argparse



class Merger(object):


    def __init__(self, argv):
        self._source = []
        self._target = []
        self._suffix = None
        self._noaction = False
        self._recurrent = False
        args = self.argument_parse(argv)
        self.update_settings(args)


    def __call__(self, argv):

        args = self.argument_parse(argv)
        self.update_settings(args)
        self.work()

    def update_settings(self, args):
        self._source = args.source
        self._target = args.target
        self._noaction = args.noaction
        self._suffix = args.suffix
        if self._suffix == None:
            self._suffix = ''
        self._recurrent = args.recurrent

    def print_settings(self):
        print('[SOURCE]:\n' + '\n'.join(self._source))

        print('<TARGET>:\n' + self._target)

        print('[SUFFIX]:\t' + str(self._suffix))

        print('[RECURRENT]:\t' + str(self._recurrent))


    def argument_parse(self, argv):
        parser = argparse.ArgumentParser(description='Merge data folders')

        parser.add_argument('--source', '-s', dest='source', nargs='+')

        parser.add_argument('--target', '-t', dest='target', default='.')
        parser.add_argument('--noaction', '-n',
                            action='store_true', default=False)

        parser.add_argument('--recurrent', '-r',
                            action='store_true', default=False)

        parser.add_argument('--suffix', dest='suffix')
        return parser.parse_args(argv)


    def refine_source(self, path):

        path = os.path.abspath(path)

        l = os.listdir(path)
        folders = [path]
        for file in l:
            pathnew = os.path.join(path, file)
            if os.path.isdir(pathnew) and self._recurrent:
                folders.extend(self.refine_source(pathnew))
        return folders


    def core_work(self, path):

        if type(path) is not str:
            raise('Core work mush get a path as input')
        path = os.path.abspath(path)
        if path == self._target:
            return None


        files = os.listdir(path)

        for file in files:

            if os.path.isdir(file):
                continue
            index = file.rfind('.')

            if index != -1:
                prefix = file[:index - 1]

                suffix = file[index + 1:]

            else:
                prefix = file
                suffix = None


            if (self._suffix is not None) and (self._suffix != suffix):

                continue
            oldfullname = os.path.join(path, file)
            newname = file
            newfullname = os.path.join(self._target, newname)

            idtmp = 1
            while os.path.isdir(newfullname) or os.path.isfile(newfullname):
                newname = prefix + '.' + str(idtmp) + '.' + suffix
                newfullname = os.path.join(self._target, newname)
                idtmp += 1
            os.rename(oldfullname, newfullname)

    def work(self):
        folders = []
        for folder in self._source:
            folders.extend(self.refine_source(folder))
        folders = list(set(folders))
        self._source = folders
        self._target = os.path.abspath(self._target)
        if self._noaction:
            self.print_settings()
        else:
            for folder in self._source:
                self.core_work(folder)

if __name__ == '__main__':

    Merger(sys.argv[1:]).work()
