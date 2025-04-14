#!/bin/bash
python scripts/active/run_eval.py \
    --model_name=<MODEL_NAME> \
    --model_path=<MODEL_PATH> \
    --save_dir=<SAVE_DIR> \
    --root_data_dir=<ROOT_DATA_DIR> \
    --dataset=<DATASET> \
    --sampling_list=<SAMPLING_LIST> \
    --split=<DATASET_SPLIT> \
    --split_type=<SPLIT_TYPE> \
    --n_targets=<NUMBER_OF_TARGETS> \
    --n_designs=<NUMBER_OF_DESIGNS> \
    --trial_length=<TRIAL_LENGTH> \
    --device=<DEVICE> \
    --peft=<PEFT> \
    --n_epochs=<NUMBER_OF_EPOCHS> \
    --seed=<SEED> \
    --proportion=<PROPORTION> 

