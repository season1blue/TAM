#! /bin/bash
python ./Train.py \
--epochs 100 \
--dataset_type 2015 \
--batch_size 8 \
--lr 2e-5 \
--text_model_name "roberta" \
--output_dir /data/results \
--output_result_file /data/result.txt \
--log_dir ./data/log.log \
--enable_log