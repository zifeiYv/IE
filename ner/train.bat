@echo on

python train.py ^
    --train_data_file train.tsv ^
    --test_data_file test.tsv ^
    --device gpu ^
    --batch_size 32 ^
    --max_seq_length 128 ^
    --logging_steps 50 ^
    --file_mode paddlenlp_mode

@cmd /k