import os
import json
from tqdm import tqdm


BASE_JSONL_FILE = "/data/user_data/peixuanh/processed_data/Anchor_DR_ALL/K_5000/train.jsonl"
TARGET_JSONL_FILE = "/data/user_data/peixuanh/processed_data/Anchor_DR_ALL/K_15000/train.jsonl"
QREL_FILE = "/data/user_data/peixuanh/data/ClueWeb22_B_ALL/K_15000/qrels.train.tsv"


groups = []

with open(QREL_FILE, "r") as file:
    for line in file:
        X = line.split('\t')
        groups.append((int)(X[-1]))

cnt = 0
with open(BASE_JSONL_FILE, "r") as fin, open(TARGET_JSONL_FILE, "w") as fout:
    for line in tqdm(fin):
        X = json.loads(line)
        X['cluster_id'] = groups[cnt]
        cnt += 1
        fout.write(json.dumps(X) + '\n')

# os.system(f"rm {BASE_JSONL_FILE}")