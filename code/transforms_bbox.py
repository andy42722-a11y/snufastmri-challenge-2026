import json

import numpy as np
import torch


def to_tensor(data):
    return torch.from_numpy(data)


class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def _boxes_for_slice(self, attrs, slice_idx):
        raw = attrs.get('annotations', '{}') if attrs is not None else '{}'
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        try:
            annotations = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            annotations = {}
        return annotations.get(str(int(slice_idx)), [])

    def _bbox_mask(self, target, boxes):
        mask = np.zeros(target.shape, dtype=np.float32)
        height, width = target.shape[-2], target.shape[-1]
        for box in boxes:
            x0 = max(0, int(box.get('x', 0)))
            y0 = max(0, int(box.get('y', 0)))
            x1 = min(width, x0 + int(box.get('width', 0)))
            y1 = min(height, y0 + int(box.get('height', 0)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
        return mask

    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            boxes = self._boxes_for_slice(attrs, slice)
            bbox_mask = self._bbox_mask(target, boxes)
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            bbox_mask = torch.from_numpy(bbox_mask)
            has_bbox = bool(boxes)
            boxes_json = json.dumps(boxes)
        else:
            target = -1
            maximum = -1
            bbox_mask = torch.zeros(1, dtype=torch.float32)
            has_bbox = False
            boxes_json = '[]'

        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice, bbox_mask, has_bbox, boxes_json
