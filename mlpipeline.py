import logging
import argparse
from src.cleaning import clean_and_save
from src.train_model import train_and_save
from src.evaluate import evaluate_slices

logging.basicConfig(level=logging.INFO)


def go(args):
    if args.step is None or args.step == "clean":
        logging.info("Started to clean data")
        clean_and_save()

    if args.step is None or args.step == "train_save_model":
        logging.info("Strated to train model")
        train_and_save()

    if args.step is None or args.step == "evaluate":
        logging.info("Evaluating model on different slices")
        evaluate_slices()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline arguments")

    parser.add_argument(
        "--step",
        type=str,
        choices=["clean", "train_save_model", "evaluate"],
        default=None,
        help="Pipeline step",
    )

    args = parser.parse_args()
    go(args)
