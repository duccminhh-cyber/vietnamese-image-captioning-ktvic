"""Train/validation EDA and batch checks; does not open official test assets."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader
from src.data.annotations import load_train_val_split, build_training_vocabulary, read_annotations
from src.data.dataset import KTVICDataset, CaptionCollator, build_transform
from src.data.tokenizer import tokenize
from src.utils.reproducibility import seed_everything


def inspect_data(data_dir, split_file):
    seed_everything(42)
    torch.set_num_threads(2)
    split = load_train_val_split(data_dir, split_file)
    raw = read_annotations(data_dir / 'train_data.json')
    vocab = build_training_vocabulary(data_dir, split, 'segment_caption')
    report = {
        'status': 'PASS', 'checked_at': datetime.now().astimezone().isoformat(),
        'scope': 'train/validation only; test counts come from frozen split metadata',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torchvision': torchvision.__version__, 'Pillow': PIL.__version__,
                     'numpy': np.__version__},
        'cuda_available': torch.cuda.is_available(),
        'source_sha256': split['source_sha256'], 'counts': split['counts'],
        'vocabulary_size': len(vocab), 'partitions': {},
    }
    by_id = {row['id']: row['filename'] for row in raw['images']}
    for name in ('train', 'val'):
        ids = set(split[f'{name}_image_ids'])
        rows = [a for a in raw['annotations'] if a['image_id'] in ids]
        tokens = [tokenize(row['segment_caption']) for row in rows]
        lengths = [len(t) for t in tokens]
        total = sum(lengths)
        missing, unreadable = [], []
        for image_id in sorted(ids):
            path = (data_dir / 'train-images' / by_id[image_id]).resolve()
            if not path.is_relative_to((data_dir / 'train-images').resolve()):
                raise ValueError('Unsafe image filename')
            if not path.is_file():
                missing.append(image_id)
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    unreadable.append({'image_id': image_id, 'error': type(exc).__name__})
        if missing or unreadable:
            raise ValueError(f'{name}: missing={missing[:10]}, unreadable={unreadable[:10]}')
        # Deterministic transforms for inspection; production train augmentation
        # remains available in build_transform('train').
        dataset = KTVICDataset(data_dir, split_file, name, vocab,
                               transform=build_transform('val'), train_val_only=True)
        batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0,
                                    collate_fn=CaptionCollator(vocab.pad_id))))
        captions, mask = batch['captions'], batch['padding_mask']
        assert batch['images'].shape == (len(captions), 3, 224, 224)
        assert torch.isfinite(batch['images']).all()
        assert torch.equal(mask, captions.eq(vocab.pad_id))
        assert torch.equal((~mask).sum(1), batch['lengths'])
        assert captions[:, 0].eq(vocab.bos_id).all()
        ends = captions.gather(1, (batch['lengths'] - 1).unsqueeze(1))
        assert ends.eq(vocab.eos_id).all()
        references = Counter(a['image_id'] for a in rows)
        report['partitions'][name] = {
            'images': len(ids), 'captions': len(rows),
            'captions_per_image': dict(Counter(references.values())),
            'empty_captions': sum(not t for t in tokens),
            'tokens_min': min(lengths), 'tokens_mean': float(np.mean(lengths)),
            'tokens_p95': float(np.percentile(lengths, 95)), 'tokens_max': max(lengths),
            'oov_token_fraction': sum(t not in vocab.token_to_id for ts in tokens for t in ts) / max(1, total),
            'missing_images': len(missing), 'unreadable_images': len(unreadable),
            'batch_images_shape': list(batch['images'].shape),
            'batch_captions_shape': list(captions.shape),
            'padding_bos_eos_checks': 'PASS',
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/week1_check.json')
    args = parser.parse_args()
    split_file = args.split_file or args.data_dir / 'splits/ktvic_seed42_content_safe.json'
    # Never overwrite dataset or project source with a generated report.
    destination = args.output.resolve()
    if destination.exists() or destination.is_relative_to(args.data_dir.resolve()):
        raise ValueError('Choose a new report path outside the data directory')
    report = inspect_data(args.data_dir, split_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
