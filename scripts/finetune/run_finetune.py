import argparse 
from pathlib import Path 
import yaml 
from finetune import train 
from finetune.args import parse_arguments

def main():
    args = parse_arguments()
    model_config_path = Path("scripts") / "finetune" / "model_args" / f"{args.model_name}.yaml"
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    for key, value in model_config.items():
        setattr(args, key, value)
    args.wandb_project = f"{args.dataset}_{args.split}"
    args.wandb_run_name = f"{args.model_name}"
    print("wandb project:", args.wandb_project)
    print("wandb run name:", args.wandb_run_name)
    if args.save_dir is not None:
        args.save_dir = Path(args.save_dir) / args.dataset / args.split / args.model_name
        args.save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.root_data_dir) / "processed" / args.dataset
    if args.split == 'entities':
        args.data_dir = data_dir / "split_entities.json"
    else:
        args.data_dir = data_dir / "split_designs.json"
    print("data_dir:", args.data_dir)
    print("save_dir:", args.save_dir)
    train(args)
    

if __name__ == "__main__":
    main()