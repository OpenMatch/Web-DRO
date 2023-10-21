import pandas as pd
import os
import csv
import sys
from tqdm import tqdm
# 读取TSV文件，手动指定列名
if len(sys.argv) > 1:
    MYPATH = sys.argv[1]
else:
    MYPATH="/data/user_data/peixuanh/data/ClueWeb22_B/web_data/res/"
tsv_file = MYPATH + 'corpus.tsv'
column_names = ['id', 'title', 'content', 'url']
df = pd.read_csv(tsv_file, sep='\t', header=None, names=column_names, quoting=csv.QUOTE_NONE)

# 将数据转换为JSONL格式
os.mkdir(MYPATH + 'jsonl/')
jsonl_output = MYPATH + 'jsonl/corpus.jsonl'
with open(jsonl_output, 'w', encoding='utf-8') as jsonl_file:
    for _, row in tqdm(df.iterrows()):
        data = {
            "id": str(row['id']),
            "title": row['title'],
            "contents": row['content']
        }
        jsonl_file.write(pd.io.json.dumps(data) + '\n')
