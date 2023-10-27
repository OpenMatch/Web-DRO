#Remember to Modify variables according to your need


torchrun --nproc_per_node=8 --master_port 19286 -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR  \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/t5  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

mkdir -p $RESULT_DIR/marco/Anchor_DR
torchrun --nproc_per_node=8 --master_port 19286 -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco/Anchor_DR \
    --model_name_or_path $CHECKPOINT_DIR/Anchor_DR/t5 \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/dev.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/marco/Anchor_DR/dev.trec  \
    --dataloader_num_workers 1 \
    --use_gpu

python scripts/evaluate.py \
    -m mrr.10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR/dev.trec

python scripts/evaluate.py \
    -m ndcg_cut_10 \
    $COLLECTION_DIR/marco/qrels_new.dev.tsv \
    $RESULT_DIR/marco/Anchor_DR/dev.trec