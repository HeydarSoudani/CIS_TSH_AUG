#!/usr/bin/env python3

### ==============================================================================
# Ref: https://github.com/kyriemao/T5QR/blob/main/utils.py
### ==============================================================================


# from IPython import embed
import os
import json
import shutil
import pickle
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def check_dir_exist_or_build(dir_list, force_emptying:bool = False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:    # not empty
            if force_emptying:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            else:
                raise FileExistsError
        else:
            continue

def json_dumps_arguments(output_path, args):   
    with open(output_path, "w") as f:
        params = vars(args)
        if "device" in params:
            params["device"] = str(params["device"])
        f.write(json.dumps(params, indent=4))

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path, high_protocol = False):
    with open(path, 'wb') as f:
        if high_protocol:  
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)


def get_has_qrel_label_sample_ids(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        qids.add(query)
    return qids

def train_dev_split_qrecc():
    with open('datasets/QReCC/qrecc_train.json', 'r') as file:
        data = json.load(file)
    
    if not isinstance(data, list):
        raise ValueError("The JSON data should be a list")
    
    random.shuffle(data)
    
    split_index = int(len(data) * 0.9)
    train_part = data[:split_index]
    dev_part = data[split_index:]
    
    with open('datasets/QReCC/new_train_qrecc.json', 'w') as train_file:
        # json.dump(train_part, train_file, indent=4)
        for item in train_part:
            train_file.write(json.dumps(item) + '\n')

    with open('datasets/QReCC/new_dev_qrecc.json', 'w') as dev_file:
        # json.dump(dev_part, dev_file, indent=4)
        for item in dev_part:
            dev_file.write(json.dumps(item) + '\n')
    



