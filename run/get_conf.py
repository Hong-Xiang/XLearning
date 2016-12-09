#!/usr/bin/python
# -*- coding: utf-8 -*-
"""script for gather necessary conf files to work dir.
"""
#TODO: update to new conf system.

raise NotImplementedError

from __future__ import absolute_import, division, print_function
import sys
import json
import os
import shutil
import xlearn.nets.model as xmod

if __name__ == "__main__":
    default_conf_path = xmod.DEFAULT_CONF_JSON
    current_dir = os.getcwd()
    net_conf = "net_conf.json"
    net_conf = os.path.join(current_dir, net_conf)
    shutil.copyfile(default_conf_path, net_conf)
    with open(net_conf, "r") as net_conf_f:
        net_confs = json.load(net_conf_f)
    net_confs_new = []
    train_file = "train_data_conf.json"
    test_file = "test_data_conf.json"
    for item in net_confs:
        if item['name'] == 'task' and len(sys.argv) > 1:
            item['value'] = sys.argv[1]
        if item['name'] == 'train_conf':
            item['value'] = os.path.join(current_dir, train_file)
        if item['name'] == 'test_conf':
            item['value'] = os.path.join(current_dir, test_file)
        net_confs_new.append(item)
    with open(net_conf, "w") as net_conf_f:
        json.dump(net_confs_new, net_conf_f, indent=4,
                  separators=[',', ': '], sort_keys=True)

    data_conf = "/home/hongxwing/Workspace/xlearn/reader/srdata_conf.json"
    shutil.copyfile(data_conf, os.path.join(current_dir, train_file))
    shutil.copyfile(data_conf, os.path.join(current_dir, test_file))
