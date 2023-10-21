#Generate embeddings using the embedding model
python -m openmatch.driver.build_index  \
    --output_dir 'TODO: path to save embeddings'  \
    --model_name_or_path 'TODO: path of the embedding model'   \
    --per_device_eval_batch_size 256  \
    --corpus_path 'TODO: path of corpus file'  \
    --doc_template "Title: <title> URL: <url> Text: <text>"  \
    --doc_column_names id,title,text,url  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

python filtering/filter_by_emb \
  'TODO: directory to embedding file(s)' \
  'TODO: directory to corpus file' \
  'TODO: name of your cluster'

#NOTICE: Please change the variables inside this python script MANUALLY.
python filtering/merge_sparse_groups.py
