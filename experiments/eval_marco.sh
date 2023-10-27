#Remember to Modify variables according to your need

torchrun --nproc_per_node=8 --master_port 19286 -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/Anchor_DR/t5  \
    --model_name_or_path $PLM_DIR/t5-ckpt  \
    --do_train  \
    --save_steps 4000  \
    --train_path /home/hanpeixuan/utilities/processed_data/Anchor_DR/train.jsonl  \
    --fp16  \
    --per_device_train_batch_size 24  \
    --train_n_passages 8  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/Anchor_DR/t5 \
    --negatives_x_device \
    --wandb_id AnchorDR

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