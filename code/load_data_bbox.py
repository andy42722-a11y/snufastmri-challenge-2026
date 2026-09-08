import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.data.transforms_bbox import DataTransform


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, lesion_sample=False, lesion_repeat=2):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []

        root = Path(root)
        kspace_files = {p.name: p for p in sorted((root / 'kspace').iterdir())}

        if forward:
            for kspace_fname in sorted(kspace_files.values()):
                num_slices = self._get_metadata(kspace_fname)
                self.examples.extend((None, kspace_fname, slice_ind) for slice_ind in range(num_slices))
            return

        image_files = sorted((root / 'image').iterdir())
        repeat = max(1, int(lesion_repeat))
        for image_fname in image_files:
            kspace_fname = kspace_files.get(image_fname.name)
            if kspace_fname is None:
                raise ValueError(f'Missing kspace pair for image file {image_fname.name}')
            num_slices = self._get_metadata(image_fname)
            annotations = self._get_annotations(image_fname)
            for slice_ind in range(num_slices):
                boxes = annotations.get(str(slice_ind), [])
                copies = repeat if lesion_sample and boxes else 1
                self.examples.extend((image_fname, kspace_fname, slice_ind) for _ in range(copies))

    def _get_metadata(self, fname):
        with h5py.File(fname, 'r') as hf:
            if self.input_key in hf.keys():
                return hf[self.input_key].shape[0]
            if self.target_key in hf.keys():
                return hf[self.target_key].shape[0]
        raise KeyError(f'Neither {self.input_key} nor {self.target_key} found in {fname}')

    def _get_annotations(self, fname):
        with h5py.File(fname, 'r') as hf:
            raw = hf.attrs.get('annotations', '{}')
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        image_fname, kspace_fname, dataslice = self.examples[i]
        with h5py.File(kspace_fname, 'r') as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf['mask'])

        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, 'r') as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, lesion_sample=False):
    if not isforward:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward,
        lesion_sample=lesion_sample,
        lesion_repeat=getattr(args, 'lesion_repeat', 2),
    )

    return DataLoader(dataset=data_storage, batch_size=args.batch_size, shuffle=shuffle)
