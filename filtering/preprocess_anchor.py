import pandas as pd
import csv
from tqdm import tqdm


documents = {}
with open("/data1/hanpeixuan/OpenMatch/data/setBA_anchor_subset-passage/corpus.tsv", 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    # 逐行读取数据并存储在列表中
    for row in tqdm(reader):
        documents[row[0]] = row[1] + '\t' + row[2]

queries = {}
with open("/data1/hanpeixuan/OpenMatch/data/setBA_anchor_subset-passage/queries.train.tsv", 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    # 逐行读取数据并存储在列表中
    for row in tqdm(reader):
        queries[row[0]] = row[1]

cnt = 0
output_file = open("/data1/hanpeixuan/OpenMatch/data/setBA_anchor_subset-passage/QDpairs.txt", "w")
with open("/data1/hanpeixuan/OpenMatch/data/setBA_anchor_subset-passage/qrels.train.tsv", 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in tqdm(reader):
        output_file.write(queries[row[0]] + '\n' + documents[row[2]] + '\n\n')
        cnt += 1