# -*- coding: utf-8 -*-
"""
Created on 2016-10-28 15:08:21

@author: HongXiang

General dataset manipulations.
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


def rename_all(folder_name, prefix=None):    
    if prefix is None:    
        l = list(str(time.time()))        
        l.remove('.')        
        t = ''.join(l)
        prefix = 'TMP'+str(t)
        listing = os.listdir(folder_name)
        for infile in listing:
            if infile[0] == '.':            
                continue
            fname_new = prefix+infile
            old_name = os.path.join(folder_name, infile)
            new_name = os.path.join(folder_name, fname_new)
            os.rename(old_name, new_name)
        else:
            rename_all(folder_name)
            listing = os.listdir(folder_name)
            cid = 0            
            for infile in listing:
                if infile[0] == '.':                    
                    continue
                index = infile.rfind('.')
                suffix = infile[index:]
                prefix_now = prefix + str(cid)
                fname_new = prefix_now+suffix
                old_name = os.path.join(folder_name, infile)
                new_name = os.path.join(folder_name, fname_new)                
                os.rename(old_name, new_name)
                cid += 1

def down_sample(src, tar, ratio=2):
    pass

def main(argv):
    parser = argparse.ArgumentParser(description='General Dataset manipulations.')
    parser.add_argument('--rename', '-r', dest='rename_folder_list')
    parser.add_argument('--merge', '-m', dest='merge_folder_list')    
    parser.add_argument('--targe', '-t', dest='targe_folder_name')

    args = parser.parse_args(argv[1:])
    #if merge is required, rename them into random names first
    for folder in args.merge_folder_list:
        rename_all(folder)
    


if __name__=="__main__":
    main(sys.argv)
    