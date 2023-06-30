#! /bin/bash
python ./Train.py \
--epochs 100 \
--dataset_type 2015 \
--batch_size 8 \
--save_step 525 \
--lr 2e-5 \
--text_model_name "roberta" \
--output_dir /data/results \
--output_result_file /data/result.txt \
--log_dir ./data/log.log \
--enable_log \
# --add_gan \
# --add_gpt \
# --add_gan_loss \

# python ./Train.py \
# --epochs 1 \
# --dataset_type 2015 \
# --batch_size 4 \
# --save_step 525 \
# --lr 2e-5 \
# --text_model_name "roberta" \
# --output_dir /data/results \
# --output_result_file /data/result.txt \
# --log_dir ./data/log.log \
# --enable_log \
# --add_gan \
# --add_gpt \
# --add_gan_loss

# python ./Train.py \
# --epochs 100 \
# --dataset_type 2015 \
# --batch_size 8 \
# --save_step 525 \
# --lr 2e-5 \
# --text_model_name "roberta" \
# --output_dir /data/results \
# --output_result_file /data/result.txt \
# --log_dir ./data/log.log \
# --enable_log \
# --add_gan \
# # --add_gpt \
# # --add_gan_loss \