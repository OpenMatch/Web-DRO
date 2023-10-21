
from concurrent.futures import ThreadPoolExecutor
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import SimpleSearcher
import pandas as pd
from tqdm import tqdm
import random
import sys
if len(sys.argv) > 1:
    MYPATH = sys.argv[1]
else:
    MYPATH="/data/user_data/peixuanh/data/ClueWeb22_B/web_data/res/"
    
searcher = LuceneSearcher(MYPATH + "index")

df = pd.read_csv(MYPATH + 'queries.train.tsv', delimiter='\t', header=None)
df = df.fillna('')
query_list = df.iloc[:,:].values.tolist()


dq_dict = {}
qid_to_text = {str(x[0]) : x[1] for x in query_list}
with open(MYPATH + 'qrels.train.tsv', "r") as file:
    for line in file:
        x = line.split('\t')
        try:
            dq_dict[x[2]].append(x[0])
        except:
            dq_dict[x[2]] = [x[0]]

print(len(query_list))
index_reader = SimpleSearcher(MYPATH + "index")
total_docs = index_reader.num_docs
print(total_docs)
cc = 0
for i in range(total_docs):
    if (str)(i) not in dq_dict:
        #print(i, end = ' ')
        dq_dict[(str)(i)] = []
        cc += 1
print(cc)

GEN_CNT = 10
def search_negatives(item):
    qid = str(item[0])
    text = item[1]
    text = " ".join(text.split()[:16])
    hits = [ins.docid for ins in searcher.search(text, k= 2 * GEN_CNT)]
    hits.extend([str(random.randrange(total_docs)) for i in range(GEN_CNT)])
    negs = ''
    cnt = 0
    #print(dq_dict[4102535])
    #print(dq_dict['7288461'])
    for i in range(len(hits[:-1])):
        # 对于q，找到的neg d必须满足两个条件
        # 条件1：这个d不是q本身所对应的d（别忘了q->d是满射但不是单射）
        # 条件2：所有对应这个d的q'里没有和q一样的（因为q没去重所以可能有这个问题）
        OK = True
        
        if qid in dq_dict[hits[i]]:
            OK = False
        for q_id in dq_dict[hits[i]]:
            try:
                if qid_to_text[q_id] == text:
                    OK = False
            except:
                print(hits[i])
                print(dq_dict[hits[i]])
                print(q_id)
                print(item)
                exit(0)
        if OK: 
            negs += str(hits[i]) + ','
            cnt += 1
            if cnt == GEN_CNT:
                break
            
    negs = negs[:-1] # 去掉最后一个逗号
    return qid, negs

# with open(MYPATH + 'train.BM25.negatives.tsvaaa', 'w', encoding='utf-8') as f:
#     for qid, negs in tqdm(map(search_negatives, query_list), total=len(query_list)):
#         pass
#         #f.write('\t'.join([qid, negs]) + '\n')
#         #exit(0)

import multiprocessing
#max_workers=24
with open(MYPATH + 'train.BM25.negatives.tsv', 'w', encoding='utf-8') as f:
    with ThreadPoolExecutor() as executor: # Adjust the number of workers as needed
        for qid, negs in tqdm(executor.map(search_negatives, query_list), total=len(query_list)):
            f.write('\t'.join([qid, negs]) + '\n')
