"""KTVIC datasets with caption-level training and image-level evaluation."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.annotations import ImageId, load_split, load_train_val_split, read_annotations
from src.data.tokenizer import CAPTION_FIELDS, Vocabulary


def build_transform(split: str, image_size: int = 224):
    """ImageNet normalization; augmentation only for the training partition."""
    if split not in ("train", "val", "test") or image_size < 32:
        raise ValueError("Expected train/val/test and image_size >= 32")
    resize = round(image_size * 256 / 224)
    operations = ([transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0),
                                               interpolation=InterpolationMode.BILINEAR),
                   transforms.RandomHorizontalFlip()]
                  if split == "train" else
                  [transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
                   transforms.CenterCrop(image_size)])
    return transforms.Compose(operations + [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class KTVICDataset(Dataset):
    """One caption per train item; image mode keeps every reference for metrics.

    In image mode, caption_ids uses reference #0 for inspecting the batch; use
    reference_captions for BLEU/CIDEr. For validation loss over every reference,
    explicitly choose sample_mode='caption'.
    """

    def __init__(self, data_dir: Path | str, split_file: Path | str, split: str,
                 vocabulary: Vocabulary, caption_field: str = "segment_caption",
                 transform=None, max_length: int | None = 64,
                 sample_mode: Literal["caption", "image"] | None = None,
                 image_dir: Path | str | None = None,
                 train_val_only: bool = False) -> None:
        if split not in ("train", "val", "test") or caption_field not in CAPTION_FIELDS:
            raise ValueError("Invalid split or caption field")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must include room for BOS and EOS")
        self.data_dir = Path(data_dir)
        if train_val_only and split == "test":
            raise ValueError("Official test is disabled for this run")
        partition = (load_train_val_split if train_val_only else load_split)(self.data_dir, split_file)
        if vocabulary.metadata.get("caption_field") != caption_field:
            raise ValueError("Vocabulary caption field does not match this dataset")
        # A serialized vocabulary from another partition must never be reused silently.
        train_id_hash = hashlib.sha256(json.dumps(sorted(partition["train_image_ids"]),
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
        if (vocabulary.metadata.get("source_sha256") != partition["source_sha256"]
                or vocabulary.metadata.get("train_image_ids_sha256") != train_id_hash
                or vocabulary.metadata.get("fit_split") != "train"):
            raise ValueError("Vocabulary was not fit on this exact training partition")
        source = read_annotations(self.data_dir / ("test_data.json" if split == "test"
                                                   else "train_data.json"))
        self.split = split
        self.vocabulary = vocabulary
        self.caption_field = caption_field
        self.max_length = max_length
        self.sample_mode = sample_mode or ("caption" if split == "train" else "image")
        if self.sample_mode not in ("caption", "image"):
            raise ValueError("sample_mode must be caption or image")
        self.transform = transform if transform is not None else build_transform(split)
        self.image_dir = (Path(image_dir) if image_dir is not None else
                          self.data_dir / ("public-test-images" if split == "test" else "train-images"))
        self.image_ids = list(partition[f"{split}_image_ids"])
        selected = set(self.image_ids)
        self.id_to_filename = {image["id"]: image["filename"] for image in source["images"]
                               if image["id"] in selected}
        self.references: dict[ImageId, list[str]] = defaultdict(list)
        annotations = []
        for annotation in source["annotations"]:
            if annotation["image_id"] in selected:
                annotations.append(annotation)
                self.references[annotation["image_id"]].append(annotation[caption_field])
        self.items = ([(a["image_id"], a[caption_field]) for a in annotations]
                      if self.sample_mode == "caption" else
                      [(image_id, self.references[image_id][0]) for image_id in self.image_ids])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        image_id, caption = self.items[index]
        filename = self.id_to_filename[image_id]
        image_path = (self.image_dir / filename).resolve()
        if not image_path.is_relative_to(self.image_dir.resolve()):
            raise ValueError(f"Unsafe image filename: {filename}")
        with Image.open(image_path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return {"images": tensor,
                "captions": torch.tensor(self.vocabulary.encode(caption, self.max_length), dtype=torch.long),
                "image_id": image_id, "filename": filename, "caption_text": caption,
                "reference_captions": list(self.references[image_id])}


class CaptionCollator:
    """Right padding; padding_mask=True means ignore this position."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, samples: list[dict]) -> dict:
        captions = pad_sequence([s["captions"] for s in samples], batch_first=True,
                                padding_value=self.pad_id)
        return {"images": torch.stack([s["images"] for s in samples]),
                "captions": captions, "lengths": torch.tensor([len(s["captions"]) for s in samples]),
                "padding_mask": captions.eq(self.pad_id),
                "image_ids": [s["image_id"] for s in samples],
                "filenames": [s["filename"] for s in samples],
                "caption_texts": [s["caption_text"] for s in samples],
                "reference_captions": [s["reference_captions"] for s in samples]}
