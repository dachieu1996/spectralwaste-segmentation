import numpy as np
from pathlib import Path

from typing import Union, Optional, Dict, Literal, Tuple

from collections import namedtuple

import torch

import torchvision
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

import imageio.v3 as imageio


class SemanticSegmentationTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.RandomRotation(30),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.ToPureTensor()
        ])

    def forward(self, *inputs):
        return self.transform(*inputs)

class SemanticSegmentationTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.ToPureTensor()
        ])

    def forward(self, *inputs):
        return self.transform(*inputs)


class SpectralWasteSegmentation(torchvision.datasets.VisionDataset):
    SpectralWasteClass = namedtuple(
        'SpectralWasteClass',
        ['name', 'id', 'color', 'ignore_in_eval']
    )

    classes = [
        SpectralWasteClass('background', 0, (0, 0, 0), True),
        SpectralWasteClass('film', 0, (218, 247, 6), False),
        SpectralWasteClass('basket', 0, (51, 221, 255), False),
        SpectralWasteClass('cardboard', 0, (52, 50, 221), False),
        SpectralWasteClass('video_tape', 0, (202, 152, 195), False),
        SpectralWasteClass('filament', 0, (0, 128, 0), False),
        SpectralWasteClass('bag', 0, (255, 165, 0), False)
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        input_mode: Union[str, list[str]] = ['rgb', 'hyper'],
        target_mode: str = 'labels_rgb',
        target_type: str = 'semantic',
        transform=None,
        target_transform=None,
        transforms=None
    ):
        super().__init__(root, transforms, transform, target_transform)

        assert target_type in ['semantic', 'instance', '']

        self.input_mode = input_mode
        self.target_mode = target_mode
        self.target_type = target_type

        if not isinstance(input_mode, list):
            self.input_mode = [input_mode]

        self.classes_names = [c.name for c in self.classes]
        self.palette = [c.color for c in self.classes]
        self.num_classes = len(self.classes_names)

        self.input_dirs = [Path(root, mode, split) for mode in self.input_mode]
        self.target_dir = Path(root, self.target_mode, split)

        self.input_paths = [list(sorted(dir.iterdir())) for dir in self.input_dirs]
        self.target_paths = list(sorted(self.target_dir.glob(f'*{target_type}.png')))

        sample = self[0]
        if isinstance(sample[0], list):
            self.num_channels = [input.shape[0] for input in sample[0]]
        else:
            self.num_channels = sample[0].shape[0]

    def __getitem__(self, idx) -> tuple[Union[torch.FloatTensor, list[torch.FloatTensor]], torch.LongTensor]:
        # load inputs
        inputs = []
        for i, m in enumerate(self.input_mode):
            path = self.input_paths[i][idx]

            if path.suffix == '.npy':
                input = np.load(path).astype(np.float32)
            elif path.suffix == '.png':
                input = imageio.imread(path)
            elif path.suffix == '.tiff':
                input = imageio.imread(path)
                input = input.transpose(1, 2, 0)
            else:
                raise ValueError

            # convert image to float if it is integer
            if issubclass(input.dtype.type, np.integer):
                input = input.astype(np.float32) / np.iinfo(input.dtype).max

            # convert to tensor
            input = tv_tensors.Image(input)
            input = input.permute(2, 0, 1)
            inputs.append(input)

        # load target
        target = imageio.imread(self.target_paths[idx])
        target = torch.tensor(target.astype(np.int64))

        if self.target_type == 'instance':
            masks = []
            labels = []
            for id in target.unique():
                if id == 0:
                    continue
                masks.append(target == id)
                labels.append(id // 1024)
            target = dict(masks=tv_tensors.Mask(torch.stack(masks)), categories=torch.stack(labels))
        elif self.target_type == 'semantic' or self.target_type == '':
            target = tv_tensors.Mask(target)
        else:
            raise ValueError

        # apply transformations
        if self.transforms:
            target, *inputs = self.transforms(target, *inputs)

        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs, target

    def __len__(self):
        return len(self.input_paths[0])


class HeterogeneousSpectralWasteSegmentation(SpectralWasteSegmentation):
    """
    Dataset wrapper for training with *heterogeneous modality availability*.

    We always load **both** modalities (RGB + HSI-like) so the model input channel
    count is constant, but optionally "drop" one modality by replacing it with
    zeros to simulate:
      - `both`: RGB + HSI available
      - `rgb`:  RGB-only (HSI is zeroed)
      - `hyper`: HSI-only (RGB is zeroed)

    Typical usage:
      - Train: `forced_mode=None` and choose per-sample mode from `modal_mix`.
      - Eval: set `forced_mode` to a fixed mode for deterministic metrics.
    """

    Mode = Literal["both", "rgb", "hyper"]
    _VALID_FORCED_MODES: Tuple[Mode, ...] = ("both", "rgb", "hyper")

    def __init__(
        self,
        root: str,
        split: str = "train",
        input_mode: Union[str, list[str]] = ["rgb", "hyper"],
        target_mode: str = "labels_rgb",
        target_type: str = "semantic",
        transform=None,
        target_transform=None,
        transforms=None,
        *,
        modal_mix: Optional[Dict[str, float]] = None,
        forced_mode: Optional[Mode] = None,
    ):
        input_modes = list(input_mode) if isinstance(input_mode, (list, tuple)) else [input_mode]
        if len(input_modes) != 2 or "rgb" not in input_modes:
            raise ValueError("HeterogeneousSpectralWasteSegmentation requires 2 inputs including 'rgb', e.g. ['rgb','hyper'] or ['rgb','hyper_pca3'].")
        other = input_modes[0] if input_modes[1] == "rgb" else input_modes[1]
        if not other.startswith("hyper"):
            raise ValueError("HeterogeneousSpectralWasteSegmentation requires the non-RGB modality to start with 'hyper' (e.g. 'hyper' or 'hyper_pca3').")

        self._rgb_idx = input_modes.index("rgb")
        self._hyper_idx = 1 - self._rgb_idx

        if forced_mode is not None and forced_mode not in self._VALID_FORCED_MODES:
            raise ValueError(f"forced_mode must be one of {list(self._VALID_FORCED_MODES)}")

        # Store mix; actual per-sample mode assignment is done after the base
        # dataset has loaded file lists (i.e. after super().__init__()).
        mix = modal_mix or {"both": 1.0, "rgb": 0.0, "hyper": 0.0}
        self._modal_mix = {
            "both": float(mix.get("both", 0.0)),
            "rgb": float(mix.get("rgb", 0.0)),
            "hyper": float(mix.get("hyper", 0.0)),
        }
        self._mode_by_idx: Optional[list[HeterogeneousSpectralWasteSegmentation.Mode]] = None

        # SpectralWasteSegmentation.__init__ probes `self[0]` to infer channel
        # counts; make this deterministic regardless of `modal_mix`.
        self.forced_mode: Optional[HeterogeneousSpectralWasteSegmentation.Mode] = "both"

        super().__init__(
            root=root,
            split=split,
            input_mode=input_modes,
            target_mode=target_mode,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.forced_mode = forced_mode

        # Assign a mode for each sample once (in init) so __getitem__ stays
        # simple and deterministic within this dataset instance.
        if self.forced_mode is None:
            probs = torch.tensor(
                [self._modal_mix["both"], self._modal_mix["rgb"], self._modal_mix["hyper"]],
                dtype=torch.float32,
            )
            total = float(probs.sum().item())
            probs = (probs / total) if total > 0 else torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            mode_ids = torch.multinomial(probs, num_samples=len(self), replacement=True)
            self._mode_by_idx = [self._VALID_FORCED_MODES[int(i)] for i in mode_ids.tolist()]

    def __getitem__(self, idx):
        inputs, target = super().__getitem__(idx)
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise RuntimeError("Expected 2-modalities inputs as a list [rgb, hyper].")

        rgb = inputs[self._rgb_idx]
        hyper = inputs[self._hyper_idx]
        if self.forced_mode is not None:
            mode = self.forced_mode
        else:
            if self._mode_by_idx is None:
                raise RuntimeError("Internal error: _mode_by_idx not initialized.")
            mode = self._mode_by_idx[idx]

        if mode == "rgb":
            hyper = torch.zeros_like(hyper)
        elif mode == "hyper":
            rgb = torch.zeros_like(rgb)

        return [rgb, hyper], target
