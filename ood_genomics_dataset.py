import json
from pathlib import Path

import torch
import numpy as np
from tfrecord.torch.dataset import MultiTFRecordDataset


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
            data_root = Path(data_root)
        self.data_root = data_root / "llr_ood_genomics"

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
        target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        self.data_transform = lambda x: self.full_transform(
            x, transform, target_transform
        )
        

    @staticmethod
    def full_transform(item, transform, target_transform):
        
        dec = np.array([int(i) for i in item["x"].decode("utf-8").split(" ")])
        x = torch.from_numpy(transform(dec.copy())).float()
        y = torch.from_numpy(target_transform(item["y"].copy())).long().squeeze()
        return x, y

    def __iter__(self):
        return map(self.data_transform, self.ds.__iter__())

if __name__ == "__main__":
    ds = OODGenomicsDataset("data", "train")
    # print("ds: ", ds.full_transform)
    # print(next(iter(ds)))
    # print(ds.label_dict)
    
