from eval.args import parse_eval_arguments
from pathlib import Path
from eval import active_qa

def main():
    args = parse_eval_arguments()
    data_dir = Path(args.root_data_dir) / "processed" / args.dataset
    if args.split_type == 'entities':
        args.data_dir = data_dir / "split_entities.json"
    else:
        args.data_dir = data_dir / "split_designs.json"
    active_qa(args)

if __name__ == "__main__":
    main()