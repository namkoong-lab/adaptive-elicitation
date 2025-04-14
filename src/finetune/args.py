import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default='data',
        help="data directory",
    )  
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["twentyq_noobj", "twentyq_obj", "opinion_qa", "eedi"],
        default='twentyq_noobj',
        help="dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["entities", "designs"],
        default='entities',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gpt2", "Llama-3.2-1B", "Llama-3.1-8B"],
        default='gpt2',
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
    )
    return parser.parse_args()