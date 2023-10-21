import csv
import pickle
from tqdm import tqdm
import os
import sys
if len(sys.argv) > 1:
    MYPATH = sys.argv[1]
else:
    MYPATH="/data/user_data/peixuanh/data/beir/trec-covid/"
    

Dict = {}
input_output_tsv_file = MYPATH + "corpus.train.tsv"
cnt = 0
#fout = open(input_output_tsv_file + "small", "w")
with open(input_output_tsv_file, "r") as file, open(input_output_tsv_file + "awa", "w") as fout:
    for line in tqdm(file):
        X = line.split('\t')
        Dict[X[0]] = cnt
        X[0] = (str)(cnt)
        X[1] = X[1].replace('"', '').replace("'", '')
        X[2] = X[2].replace('"', '').replace("'", '')
        if len(X[2]) > 10000:
            X[2] = X[2][:10000] + '\n'
        fout.write('\t'.join(X))
        cnt = cnt + 1

pickle.dump(Dict, open(MYPATH + "doc_id_mapping.pkl", "wb"))
os.system(f"rm {input_output_tsv_file}")
os.system(f"mv {input_output_tsv_file + 'awa'} {input_output_tsv_file}")
print(cnt)

#input_output_tsv_file = "/home/peixuanh/1.tsv"
cnt = 0
Dict = pickle.load(open(MYPATH + "doc_id_mapping.pkl", "rb"))
input_tsv_file = MYPATH + "qrels.train.tsv"
with open(input_tsv_file , "r") as file, open(input_tsv_file + "awa", "w") as fout:
    for line in tqdm(file):
        X = line.split('\t')
        X[2] = str(Dict[X[2]])
        #print(X[2])
        fout.write('\t'.join(X))

os.system(f"rm {input_tsv_file}")
os.system(f"mv {input_tsv_file + 'awa'} {input_tsv_file}")