from argparse import ArgumentParser
from pathlib import Path

from tfrecord.tools import tfrecord2idx


if __name__ == "__main__":
    parser = ArgumentParser("Create .index files for .tfrecord files in the dataset.")
    parser.add_argument("--dataset_root", type=str, required=True)

    args = parser.parse_args()

    # Dataset from https://github.com/google-research/google-research/tree/master/genomics_ood
    DATASET_ROOT = Path(args.dataset_root)
    assert DATASET_ROOT.is_dir(), "Provided dataset root does not exist."

    id_train_path = DATASET_ROOT / "before_2011_in_tr"
    id_val_path = DATASET_ROOT / "between_2011-2016_in_val"
    id_test_path = DATASET_ROOT / "after_2016_in_test"

    ood_val_path = DATASET_ROOT / "between_2011-2016_ood_val"
    ood_test_path = DATASET_ROOT / "after_2016_ood_test"

    splits = {
        "train": id_train_path,
        "val": id_val_path,
        "test": id_test_path,
        "val_ood": ood_val_path,
        "test_ood": ood_test_path
    }

    for split, split_path in splits.items():
        tf_record_files = [f for f in split_path.iterdir() if f.suffix == ".tfrecord"]
        for tf_record_file in tf_record_files:
            tf_index_file = tf_record_file.parent / (tf_record_file.stem + ".index")
            tfrecord2idx.create_index(str(tf_record_file), str(tf_index_file))
