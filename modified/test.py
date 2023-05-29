import random
import os
import json

import glob
import io
import pyarrow as pa
from PIL import Image

import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        names = ['mmimdb_dev', 'mmimdb_test', 'mmimdb_train']

        tables = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{data_dir}/{name}.arrow", "r")
            ).read_all()
            for name in names
            if os.path.isfile(f"{data_dir}/{name}.arrow")
        ]
        table_names = list()
        for i, name in enumerate(names):
            table_names += [name] * len(tables[i])

        self.table = pa.concat_tables(tables, promote=True)

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        txt = self.table[idx]

        img = None ## TODO

        return txt, img

if __name__ == '__main__':

    data_dir = '/home/yunjinna/missing/datasets/mmimdb/'
    names = ['mmimdb_dev', 'mmimdb_test', 'mmimdb_train']

    dataset = MyDataset(
        data_dir=data_dir,
        # transform_keys=
    )

    print(len(dataset))

    exit()

    tables = [
        pa.ipc.RecordBatchFileReader(
            pa.memory_map(f"{data_dir}/{name}.arrow", "r")
        ).read_all()
        for name in names
        if os.path.isfile(f"{data_dir}/{name}.arrow")
    ]
    table_names = list()
    for i, name in enumerate(names):
        table_names += [name] * len(tables[i])

    table = pa.concat_tables(tables, promote=True)
    print(table[:1])
    exit()

    files = os.listdir(path)
    # print(files[:5])

    txt = f"{path}/list.txt"
    with open(txt, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    data = [data[idx].split('.')[1] for idx in range(len(data))]

    jsons = glob.glob('/home/yunjinna/missing/datasets/mmimdb/mmimdb/dataset/*.json')
    # print(jsons[:5])

    sample = jsons[0]
    with open(sample, 'r', encoding='utf-8') as f:
        sample_content = json.load(f)
    
    print(sample_content.keys())
    print(sample_content['music department'])

