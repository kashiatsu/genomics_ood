import json
from pathlib import Path
from numpy.core.numeric import full

import torch
from tfrecord.torch.dataset import MultiTFRecordDataset


def full_transform(item, transform, target_transform):
    x = torch.from_numpy(transform(item["x"].copy()))
    y = torch.from_numpy(target_transform(item["y"].copy()))
    return x, y


class OODGenomicsDataset(torch.utils.data.IterableDataset):
    """PyTorch Dataset implementation for the Bacteria Genomics OOD dataset (https://github.com/google-research/google-research/tree/master/genomics_ood) proposed in

    J. Ren et al., “Likelihood Ratios for Out-of-Distribution Detection,” arXiv:1906.02845 [cs, stat], Available: http://arxiv.org/abs/1906.02845.
    """

    splits = {
        "train": "before_2011_in_tr",
        "val": "between_2011-2016_in_val",
        "test": "after_2016_in_test",
        "val_ood": "between_2011-2016_ood_val",
        "test_ood": "after_2016_ood_test",
    }

    def __init__(self, data_root, split="train", transform=None, target_transform=None):
        if isinstance(data_root, str):
            self.data_root = Path(data_root)

        assert split in self.splits, f"Split '{split}' does not exist."
        split_dir = self.data_root / self.splits[split]

        tf_record_ids = [f.stem for f in split_dir.iterdir() if f.suffix == ".tfrecord"]

        self.ds = MultiTFRecordDataset(
            data_pattern=str(split_dir / "{}.tfrecord"),
            index_pattern=str(split_dir / "{}.index"),
            splits={id_: 1 / len(tf_record_ids) for id_ in tf_record_ids},
            description={"x": "byte", "y": "int", "z": "byte"},
        )

        with open(self.data_root / "label_dict.json") as f:
            label_dict = json.load(f)
            self.label_dict = {v: k for k, v in label_dict.items()}

        transform = transform if transform is not None else lambda x: x 
        target_transform = target_transform if target_transform is not None else lambda x: x 
        self.full_transform = lambda x: full_transform(x, transform, target_transform)

    def __iter__(self):
        return map(self.full_transform, self.ds.__iter__())


if __name__ == "__main__":
    ds = OODGenomicsDataset("./data/llr_ood_genomics", "train")
    print(next(iter(ds)))
    print(ds.label_dict)
