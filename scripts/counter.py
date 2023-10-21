import json
from tqdm import tqdm
import torch
MYPATH = "/data/user_data/peixuanh/processed_data/Anchor_DR_ALL/cont_t5_major/"
counter = {}
with open(MYPATH + "train.jsonl", "r") as file:
    for line in tqdm(file):
        dict = json.loads(line)
        id = (int)(dict['cluster_id'])
        try:
            counter[id] += 1
        except:
            counter[id] = 1

sorted_values = [value for key, value in sorted(counter.items())]
print(len(sorted_values))
torch.save(torch.tensor(sorted_values), MYPATH + "counter.pt")