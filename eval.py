import argparse
from config.build_config import build_config
from models.builder_model import build_model
from evaluator.build_evaluator import build_evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', default="RSN", type=str, help='train config file path')
    args = parser.parse_args()
    cfg = build_config(args.config)
    model = build_model(cfg.model)
    evaluator = build_evaluator(cfg.eval, model)
    result = evaluator.eval()
    print(result)

