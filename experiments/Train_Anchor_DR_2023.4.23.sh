export BASE_DIR=
export COLLECTION_DIR=$BASE_DIR/data
export PROCESSED_DIR=$BASE_DIR/processed_data
export PLM_DIR=$BASE_DIR/models
export CHECKPOINT_DIR=$BASE_DIR/ckpts
export LOG_DIR=$BASE_DIR/log
export EMBEDDING_DIR=$BASE_DIR/embedding
export RESULT_DIR=$BASE_DIR/res
export CUDA_VISIBLE_DEVICES=
export LOCAL_RANK=

python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/web_continuous_t5_ckpt  \
    --negative_file $COLLECTION_DIR/setBA_anchor_subset-passage/train.BM25.negatives.tsv  \
    --qrels $COLLECTION_DIR/setBA_anchor_subset-passage/qrels.train.tsv  \
    --queries $COLLECTION_DIR/setBA_anchor_subset-passage/queries.train.tsv  \
    --collection $COLLECTION_DIR/setBA_anchor_subset-passage/corpus.tsv  \
    --save_to $PROCESSED_DIR/Anchor_DR/cont_t5/  \
    --doc_template "Title: <title> Text: <text>"

    
python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/web_continuous_t5_ckpt  \
    --negative_file $COLLECTION_DIR/setBA_anchor_subset-passage/train.BM25.negatives.tsv  \
    --qrels $COLLECTION_DIR/setBA_anchor_subset-passage/qrels.train.tsv  \
    --queries $COLLECTION_DIR/setBA_anchor_subset-passage/queries.train.tsv  \
    --collection $COLLECTION_DIR/setBA_anchor_subset-passage/corpus.tsv  \
    --save_to $PROCESSED_DIR/Anchor_DR/cont_t5/  \
    --with_url \
    --doc_template "Title: <title> URL: <url> Text: <text>"
#或许在用不同数据集时应该先执行rm -rf /home/hanpeixuan/.cache/huggingface/datasets/csv

cat $PROCESSED_DIR/Anchor_DR/cont_t5/split*.jsonl > $PROCESSED_DIR/Anchor_DR/cont_t5/train.jsonl
rm $PROCESSED_DIR/Anchor_DR/cont_t5/split*.jsonl


for file in $PROCESSED_DIR/Anchor_DR/cont_t5/split*.jsonl; do
  cat "$file" >> $PROCESSED_DIR/Anchor_DR/cont_t5/train.jsonl
  rm "$file"
done

#训练
#注意device有4个
#warmup是默认0.1的
#注意nproc_per_node要改
python -m torch.distributed.launch --nproc_per_node=4 --use-env --master_port 19286 -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --model_name_or_path $PLM_DIR/web_continuous_t5_ckpt  \
    --do_train  \
    --save_steps 6000  \
    --train_path $PROCESSED_DIR/Anchor_DR/cont_t5/train.jsonl  \
    --fp16  \
    --per_device_train_batch_size 32  \
    --train_n_passages 8  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/Anchor_DR/cont_t5_ver3 \
    --negatives_x_device


#在MsMarco上inference并测试
python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR_t5_ver3  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1


mkdir $RESULT_DIR/marco/Anchor_DR_t5_ver3
python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR_t5_ver3  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/dev.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/marco/Anchor_DR_t5_ver3/dev.trec  \
    --dataloader_num_workers 1 \
    --use_gpu

python scripts/evaluate.py \
    -m mrr.10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR_t5_ver3/dev.trec

python scripts/evaluate.py \
    -m ndcg_cut_10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR_t5_ver3/dev.trec

#在AnchorDR上上inference并生成hard negative
python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/Anchor_DR/cont_t5_ver3/  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/setBA_anchor_subset-passage/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

mkdir $RESULT_DIR/setBA_anchor_subset-passage/cont_t5_ver3/
python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/Anchor_DR/cont_t5_ver3/  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/setBA_anchor_subset-passage/queries.train.tsv  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/setBA_anchor_subset-passage/cont_t5_ver3/train.trec  \
    --dataloader_num_workers 1 \
    --use_gpu

mkdir $PROCESSED_DIR/Anchor_DR/cont_t5_ver3
python ./scripts/msmarco/build_hn.py  \
    --tokenizer_name $PLM_DIR/web_continuous_t5_ckpt  \
    --hn_file $RESULT_DIR/setBA_anchor_subset-passage/cont_t5_ver3/train.trec  \
    --qrels $COLLECTION_DIR/setBA_anchor_subset-passage/qrels.train.tsv  \
    --queries $COLLECTION_DIR/setBA_anchor_subset-passage/queries.train.tsv  \
    --collection $COLLECTION_DIR/setBA_anchor_subset-passage/corpus.tsv  \
    --save_to $PROCESSED_DIR/Anchor_DR/cont_t5_ver3  \
    --doc_template "Title: <title> Text: <text>"


for file in $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/split*.hn.jsonl; do
    cat "$file" >> $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/train.hn.jsonl
    rm "$file"
done

cat $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/split*.hn.jsonl > $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/train.hn.jsonl
rm $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/split*.hn.jsonl

#用hard negative训练
python -m torch.distributed.launch --nproc_per_node=4 --use-env --master_port 19286 -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/Anchor_DR/cont_t5_s2_ver3  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_ver3  \
    --do_train  \
    --save_steps 6000  \
    --train_path $PROCESSED_DIR/Anchor_DR/cont_t5_ver3/train.hn.jsonl  \
    --fp16  \
    --per_device_train_batch_size 32  \
    --train_n_passages 8  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/Anchor_DR/cont_t5_s2_ver3 \
    --negatives_x_device

#在MsMarco上inference并测试
python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR_t5_s2_ver3  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_s2_ver3  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

mkdir $RESULT_DIR/marco/Anchor_DR_t5_s2_ver3
python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR_t5_s2_ver3  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/cont_t5_s2_ver3  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/dev.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/marco/Anchor_DR_t5_s2_ver3/dev.trec  \
    --dataloader_num_workers 1 \
    --use_gpu

python scripts/evaluate.py \
    -m mrr.10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR_t5_s2_ver3/dev.trec

python scripts/evaluate.py \
    -m ndcg_cut_10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR_t5_s2_ver3/dev.trec



#多机
torchrun --nproc_per_node=8  --master_port=$MASTER_PORT \
    --node_rank=1 --nnodes=2 --master_addr=$MASTER_IP -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/ClueWeb22_All/web_graph  \
    --model_name_or_path $PLM_DIR/web_continuous_t5_ckpt  \
    --do_train  \
    --save_steps 10000  \
    --train_path /data/user_data/peixuanh/processed_data/Anchor_DR_ALL/web_graph/train.jsonl.clean \
    --fp16  \
    --per_device_train_batch_size 16 \
    --train_n_passages 8  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/ClueWeb22_Max/web_graph \
    --negatives_x_device \
    --wandb_id web_graph