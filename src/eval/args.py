import os 
import argparse 
import torch

def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default='gpt2',
        help="model name",
        choices=["gpt2", "Llama-3.2-1B", "Llama-3.1-8B"],
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default='/scratch/jw4209/language_uncertainty/twentyq/gpt2/mask_False',
    )
    parser.add_argument(
        "--sampling_list",
        type=str,
        nargs='+',
        default=['random'],
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default='data',
    )
    parser.add_argument(
        "--peft",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='twentyq_noobj',
        choices=["twentyq_noobj", "twentyq_obj", "opinion_qa", "eedi"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default='test',
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default='entities',
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n_designs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--trial_length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='results_calib',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--proportion",
        type=float,
        default=None,
    )
    parsed_args = parser.parse_args()
    return parsed_args