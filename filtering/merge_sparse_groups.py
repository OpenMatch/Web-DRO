import os
import json
from tqdm import tqdm
import torch

QREL_FILE = "/data/user_data/peixuanh/data/ClueWeb22_B_ALL/K_15000/qrels.train.tsv"
COUNTER_FILE = "/data/user_data/peixuanh/data/ClueWeb22_B_ALL/K_15000/counter.pt"

counter = torch.load(COUNTER_FILE).tolist()
mapping = {}

valid_group_cnt = 0
err = 0
cnt = 0
for val in counter:
    if val >= 128:
        valid_group_cnt += 1
for (id, val) in enumerate(counter):
    if val >= 128:
        mapping[id] = cnt
        cnt += 1
    else:
        err += val
        mapping[id] = valid_group_cnt
print(valid_group_cnt)
print(err)

if err < 128:
    for key in mapping:
        if mapping[key] == valid_group_cnt:
            mapping[key] = 0
    valid_group_cnt -= 1
        
with open(QREL_FILE, "r") as file:
    A = file.readlines()

counter = []
for i in range(valid_group_cnt + 1):
    counter.append(0)

with open(QREL_FILE + 'z', "w") as file:
    for line in A:
        X = line.split('\t')
        X[4] = (str)(mapping[(int)(X[4])])
        counter[(int)(X[4])] += 1
        file.write('\t'.join(X) + '\n')
        
#print(counter)
print(len(counter))
torch.save(torch.tensor(counter), COUNTER_FILE + 'z')