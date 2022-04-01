from pathlib import Path

DATASET_NAMES = dict(motor_adaptation="ECs_E50")

def _get_master_dataset_location():
    """Handles finding the source data of the analysis by reading the repo
    dataset_location.txt file.

    """
    specification_txt = Path(__file__).parent.parent / "dataset_location.txt"
    if specification_txt.exists():
        with open(specification_txt, "r") as f:
            return Path(f.read())
    else:
        raise FileNotFoundError("dataset_location.txt file not found in project folder!")


def get_dataset_location(dataset_name):
    return _get_master_dataset_location() / DATASET_NAMES[dataset_name]