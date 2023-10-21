export CUDA_VISIBLE_DEVICES='TODO'
export LOCAL_RANK='TODO'

###Generate negatives by BM25
python scripts/convert_tsv_to_jsonl.py \
  'TODO: directory of corpus file'

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input 'TODO: directory of corpus file'/jsonl \
  --index 'TODO: directory of corpus file'/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 32 -storePositions --storeDocvectors --storeRaw

python scripts/generate_negative.py \
    'TODO: directory of corpus file'

#Generate the training file
python scripts/msmarco/build_train.py \
    --tokenizer_name 'TODO: path of the base model'  \
    --negative_file 'TODO: path of negative file'  \
    --qrels 'TODO: path of qrel file'  \
    --queries 'TODO: path of query file'  \
    --collection 'TODO: path of corpus file'  \
    --save_to 'TODO: place to save train.jsonl file'  \
    --doc_template "Title: <title> URL: <url> Text: <text>"

for file in 'TODO: place to save train.jsonl file'/split*.jsonl; do
  cat "$file" >> 'TODO: place to save train.jsonl file'/train.jsonl
  rm "$file"
done


#Train
python -m torchrun --nproc_per_node=8 --master_port 19287 -m openmatch.driver.train_DRO  \
    --output_dir 'TODO: place to save your model'  \
    --model_name_or_path 'TODO: path of the base model'  \
    --do_train  \
    --save_steps 6000  \
    --train_path 'TODO: place to save train.jsonl file'  \
    --fp16  \
    --per_device_train_batch_size 32 \
    --train_n_passages 8  \
    --learning_rate 3e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --negatives_x_device \
    --use_dro \
    --alpha_log_interval 500 \
    --alpha_lr 3e-4 \
    --train_alpha \
    --mixed_dataset \
    --distribution_counter 'TODO: path of the counter file'


#Generate hard negatives
python -m openmatch.driver.build_index  \
    --output_dir 'TODO: path to save embeddings'  \
    --model_name_or_path 'TODO: path of your model'   \
    --per_device_eval_batch_size 256  \
    --corpus_path 'TODO: path of corpus file'  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

mkdir $RESULT_DIR/setBA_anchor_subset-passage/cont_t5_ver3/
python -m openmatch.driver.retrieve  \
    --output_dir 'TODO: path to save embeddings'  \
    --model_name_or_path 'TODO: path of your model'   \
    --per_device_eval_batch_size 256  \
    --query_path 'TODO: path of query file'  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path 'TODO: path to save trec file'  \
    --dataloader_num_workers 1 \
    --use_gpu

python ./scripts/msmarco/build_hn.py  \
    --tokenizer_name 'TODO: path of the base model'  \
    --hn_file 'TODO; path of the trec file'  \
    --qrels 'TODO: path of qrel file'  \
    --queries 'TODO: path of query file'  \
    --collection 'TODO: path of corpus file'  \
    --save_to 'TODO: place to save train.hn.jsonl file'  \
    --doc_template "Title: <title> Text: <text>"


for file in 'TODO: place to save train.hn.jsonl file'/split*.hn.jsonl; do
    cat "$file" >> 'TODO: place to save train.hn.jsonl file'/train.hn.jsonl
    rm "$file"
done

#ANCE
python -m torchrun --nproc_per_node=8 --master_port 19287 -m openmatch.driver.train_DRO  \
    --output_dir 'TODO: place to save your ANCE model'  \
    --model_name_or_path 'TODO: place of your model'  \
    --do_train  \
    --save_steps 6000  \
    --train_path 'TODO: place to save train.hn.jsonl file'  \
    --fp16  \
    --per_device_train_batch_size 32 \
    --train_n_passages 8  \
    --learning_rate 3e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --negatives_x_device \
    --use_dro \
    --alpha_log_interval 500 \
    --alpha_lr 3e-4 \
    --train_alpha \
    --mixed_dataset \
    --distribution_counter 'TODO: path of the counter file'

