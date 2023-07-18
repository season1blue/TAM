#! /bin/bash
python3 -X faulthandler ./Train.py \
--epochs 100 \
--save_steps 500 \
--dataset_type 2015 \
--batch_size 8 \
--lr 2e-5 \
--text_model_name "flant5" \
--output_dir /data/results \
--output_result_file /data/result.txt \
--log_dir ./data/log.log \
--device_id "cuda:1" \
--enable_log \
--only_text_loss \
# --add_gan \
# --add_gan_loss
# --alpha 0 \
# --beta 0 \
# --add_llm \
