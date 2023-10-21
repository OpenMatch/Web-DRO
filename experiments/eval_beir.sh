set -e

model_name=$1

export OUTPUT_DIR='TODO: place to output results'
export model_name_or_path="TODO: father directory of your models"/$model_name
export DATA_DIR="TODO: path of BEIR"

export CUDA_VISIBLE_DEVICES="TODO"
export LOCAL_RANK="TODO"
PORT = 12345
fi
dataset_name_list=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever trec-news signal1m bioasq robust04)

#IF YOU WANT TO EVALUATE cqadupstack, UNCOMMENT THE FOLLOWING LINES.
# export OUTPUT_DIR=$BASE_DIR/res/beir/$model_name/cqadupstack
# export DATA_DIR=$BASE_DIR/data/beir/cqadupstack
# dataset_name_list=(android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress) 

mkdir -p ${OUTPUT_DIR}

for dataset_name in ${dataset_name_list[@]}
do
    export infer_job_name=inference.${dataset_name}
    
    if [ ${dataset_name} == fiqa ] || [ ${dataset_name} == signal1m ] || [ ${dataset_name} == quora ] || [ ${dataset_name} == trec-covid ] 
    then
        export q_max_len=64
        export p_max_len=128
    elif [ ${dataset_name} == scifact ] || [ ${dataset_name} == trec-news ] || [ ${dataset_name} == robust04 ]
    then
        export q_max_len=64
        export p_max_len=256
    else
        export q_max_len=128
        export p_max_len=128
    fi

    if [ ${dataset_name} == trec-covid ] || [ ${dataset_name} == trec-news ] || [ ${dataset_name} == bioasq ]
    then
        doc_temp="Title: <title> Text: <text>"
    elif [ ${dataset_name} == arguana ]
    then
        doc_temp="<text>"
    elif [ ${dataset_name} == signal1m ] || [ ${dataset_name} == robust04 ] || [ ${dataset_name} == webis-touche2020 ]
    then
        doc_temp="Text: <text>"
    else
        doc_temp="<title> <text>"
    fi

    doc_temp="Title: Text: <text>"

    echo ${infer_job_name}
    
    python -m  torch.distributed.launch --use-env --nproc_per_node=4 --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
    --data_dir ${DATA_DIR}/${dataset_name} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${OUTPUT_DIR}/${infer_job_name} \
    --query_template "<text>" \
    --doc_template "$doc_temp" \
    --q_max_len ${q_max_len} \
    --p_max_len ${p_max_len}  \
    --per_device_eval_batch_size 256  \
    --dataloader_num_workers 1 \
    --fp16 \
    --use_gpu \
    --overwrite_output_dir \
    --use_split_search \
    --max_inmem_docs 5000000

    rm -rf $OUTPUT_DIR/$infer_job_name
    echo "Job Finished: $infer_job_name"

done