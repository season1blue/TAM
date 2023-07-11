#! /bin/bash
python3 -X faulthandler ./Train.py \
--epochs 100 \
--save_steps 500 \
--dataset_type 2015 \
--batch_size 8 \
--lr 2e-5 \
--text_model_name "roberta" \
--output_dir /data/results \
--output_result_file /data/result.txt \
--log_dir ./data/log.log \
--enable_log \
# --add_llm \
# --alpha 0 \
# --beta 0 \
# --add_gan \
# --add_gan_loss
