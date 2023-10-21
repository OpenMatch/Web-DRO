import pickle
from urllib.parse import urlparse
from tqdm import tqdm
import sys
import os
import torch
def extract_main_domain(url):
    # 使用urlparse函数解析URL
    parsed_url = urlparse(url)
    # 提取主域名
    main_domain = parsed_url.netloc
    # 如果主域名以www开头，去掉www
    if main_domain.startswith('www.'):
        main_domain = main_domain[4:]
    # if '.' in main_domain:
    #     main_domain = main_domain[:main_domain.find('.')]
    return main_domain


cnt = 0
url_dict = {}

if len(sys.argv) > 1:
    MYPATH = sys.argv[1]
else:
    MYPATH="/data/user_data/peixuanh/data/ClueWeb22_Max/"
OUTPATH = MYPATH + "url_128/"
os.system(f"mkdir {OUTPATH}")

with open(MYPATH + "corpus.tsv", 'r') as f:
    for line in f:
        x = line.split('\t')
        cnt += 1
        domain = extract_main_domain(x[-1])
        try:
            url_dict[domain].append(x[0])
        except:
            url_dict[domain] = [x[0]]
            
            
sorted_urls = sorted(url_dict.items(), key=lambda item: -len(item[1]))
major_url_domains = {}
total = 0
for i in range(10000000000):
    total += len(sorted_urls[i][1])
    if(len(sorted_urls[i][1]) < 128):
        break
    #print(sorted_urls[i])
    major_url_domains[sorted_urls[i][0]] = i
pickle.dump(major_url_domains, open(f"{OUTPATH}major_url_domains.pkl", "wb"))
print(len(major_url_domains))
print(total)
print(cnt)

# major_url_domains = pickle.load(open(f"{MYPATH}major_url_domains.pkl", "rb"))

doc_id_to_url = {}
with open(MYPATH + "corpus.tsv", 'r') as f:
    for line in f:
        X = line.split('\t')
        doc_id_to_url[X[0]] = extract_main_domain(X[-1])

print("START")
qrel_path = MYPATH + "qrels.train.tsv"
qrel_out_path = OUTPATH + "qrels.train.tsv"
valid_query_ids = []
FINAL_GROUP_ID = len(major_url_domains)

group_counter = []
for i in range(len(major_url_domains) + 1):
    group_counter.append(0)

with open(qrel_path, "r") as fin, open(qrel_out_path, "w") as fout:
    for line in tqdm(fin):
        X = line.split('\t')
        if doc_id_to_url[X[2]] in major_url_domains:
            group_counter[major_url_domains[doc_id_to_url[X[2]]]] += 1
            fout.write(line[:-1] + f'\t{major_url_domains[doc_id_to_url[X[2]]]}\n')
        else:
            group_counter[FINAL_GROUP_ID] += 1
            fout.write(line[:-1] + f'\t{FINAL_GROUP_ID}\n')
            
torch.save(torch.tensor(group_counter), f"{OUTPATH}counter.pt")
# 128:
# 4731
# 4113656
# 8098771
