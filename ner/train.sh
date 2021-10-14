python train.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --device gpu \
    --batch_size 4 \
    --max_seq_length 128 \
    --logging_steps 50 \
    --train_data_file /Volumes/工作/2021年日常/7-北京业扩报装项目/InformationExtraction/ner/custom_datasets/only_train.txt \
    --file_mode typical_mode \
    --output_dir ./tmp/msra_ner/