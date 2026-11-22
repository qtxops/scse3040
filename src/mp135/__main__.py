import argparse
import json

from mp135.config import load_config
from mp135.pipeline import prepare_data, regenerate_predictions, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="MP135 data audit, training and predictions")
    parser.add_argument("stage", choices=["audit", "run", "predict"], default="run", nargs="?")
    args = parser.parse_args()
    config = load_config()
    if args.stage == "audit":
        _, _, _, issues, _ = prepare_data(config)
        print(json.dumps(issues, indent=2))
    elif args.stage == "predict":
        regenerate_predictions(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
