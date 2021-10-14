@echo off

start python train.py ^
    --model_name_or_path bert-base-multilingual-uncased ^
    --device gpu ^
    --batch_size 4 ^
    --max_seq_length 128 ^
    --logging_steps 50 ^
    --train_data_file C:\User\lilei\Desktop\InformationExtraction\ner\custom_datasets\only_train.txt ^
    --file_mode typical_mode ^
    --output_dir C:\User\lilei\Desktop\InformationExtraction\ner\custom_models