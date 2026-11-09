"""Catch split leakage, vocabulary leakage, truncation, and mask mistakes."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from src.data.annotations import (build_training_vocabulary, create_content_safe_split, create_split,
                                 load_split, read_annotations, sha256_file, validate_partition)
from src.data.audit import extract_archive_once, image_audit
from src.data.dataset import CaptionCollator, KTVICDataset, build_transform
from src.data.tokenizer import Vocabulary, tokenize


class TokenizerTests(unittest.TestCase):
    def test_normalization_and_segment_preservation(self):
        self.assertEqual(tokenize("  MỘT\tcon_thuyền  "), ["một", "con_thuyền"])

    def test_stable_ids_truncation_and_unknown(self):
        first = Vocabulary.build(["b a b", "c a"], min_frequency=2)
        second = Vocabulary.build(["c a", "b a b"], min_frequency=2)
        self.assertEqual(first.id_to_token, second.id_to_token)
        self.assertEqual(first.id_to_token[:4], ["<pad>", "<bos>", "<eos>", "<unk>"])
        self.assertEqual(first.encode("a b c", max_length=3), [first.bos_id, first.token_to_id["a"], first.eos_id])
        self.assertEqual(first.encode("unknown"), [first.bos_id, first.unk_id, first.eos_id])
        self.assertEqual(first.decode(first.encode("a unknown")), "a <unk>")
        with self.assertRaises(ValueError):
            first.encode("a", max_length=1)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for source, ids, directory in (("train", range(1, 9), "train-images"),
                                       ("test", (100, 101), "public-test-images")):
            (self.root / directory).mkdir()
            images, annotations = [], []
            for image_id in ids:
                filename = f"{image_id}.png"
                images.append({"id": image_id, "filename": filename})
                Image.new("RGB", (48, 40), (image_id % 255, 40, 80)).save(self.root / directory / filename)
                for reference in range(2):
                    text = f"ảnh chung unique{image_id} ref{reference}"
                    annotations.append({"id": len(annotations), "image_id": image_id,
                                        "caption": text, "segment_caption": text})
            (self.root / f"{source}_data.json").write_text(
                json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
        self.split_file = self.root / "splits" / "fixture.json"
        self.split = create_split(self.root, self.split_file, validation_fraction=.25)

    def tearDown(self):
        self.temporary.cleanup()

    def test_split_reproducibility_counts_and_ref_preservation(self):
        repeated = create_split(self.root, self.split_file, validation_fraction=.25)
        self.assertEqual(self.split, repeated)
        self.assertEqual(self.split["counts"]["train"], {"images": 6, "annotations": 12})
        self.assertEqual(self.split["counts"]["val"], {"images": 2, "annotations": 4})
        with self.assertRaises(ValueError):
            create_split(self.root, self.split_file, seed=7, validation_fraction=.25)

    def test_reject_leakage_and_orphans(self):
        train, test = [read_annotations(self.root / f"{source}_data.json") for source in ("train", "test")]
        leaked = deepcopy(self.split)
        leaked["val_image_ids"].append(leaked["train_image_ids"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_partition(train, test, leaked)
        orphaned = deepcopy(train)
        orphaned["annotations"][0]["image_id"] = 9999
        with self.assertRaisesRegex(ValueError, "Orphan"):
            validate_partition(orphaned, test, self.split)

    def test_changed_raw_json_is_detected(self):
        source = self.root / "train_data.json"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            load_split(self.root, self.split_file)

    def test_vocabulary_never_sees_val_or_test(self):
        vocabulary = build_training_vocabulary(self.root, self.split, "caption", min_frequency=1)
        for image_id in self.split["train_image_ids"]:
            self.assertIn(f"unique{image_id}", vocabulary.token_to_id)
        for image_id in self.split["val_image_ids"] + self.split["test_image_ids"]:
            self.assertNotIn(f"unique{image_id}", vocabulary.token_to_id)

    def test_dataset_modes_and_all_references(self):
        vocabulary = build_training_vocabulary(self.root, self.split, "caption", min_frequency=1)
        for split in ("train", "val", "test"):
            dataset = KTVICDataset(self.root, self.split_file, split, vocabulary, "caption",
                                   transform=build_transform(split, 32))
            self.assertEqual(len(dataset), self.split["counts"][split]["annotations" if split == "train" else "images"])
            item = dataset[0]
            self.assertEqual(item["images"].shape, (3, 32, 32))
            self.assertEqual(len(item["reference_captions"]), 2)
        dataset = KTVICDataset(self.root, self.split_file, "val", vocabulary, "caption", sample_mode="caption")
        self.assertEqual(len(dataset), 4)
        wrong = Vocabulary(vocabulary.id_to_token, {**vocabulary.metadata, "fit_split": "test"})
        with self.assertRaisesRegex(ValueError, "not fit"):
            KTVICDataset(self.root, self.split_file, "train", wrong, "caption")

    def test_archive_path_validation_and_idempotent_extraction(self):
        import zipfile
        unsafe = self.root / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../escape.txt", "unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            extract_archive_once(unsafe, self.root)
        safe = self.root / "safe.zip"
        with zipfile.ZipFile(safe, "w") as archive:
            archive.writestr("extracted/example.txt", "original")
        self.assertEqual(extract_archive_once(safe, self.root)["extracted"], 1)
        self.assertEqual(extract_archive_once(safe, self.root)["existing_identical"], 1)
        (self.root / "extracted" / "example.txt").write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing|CRC differs"):
            extract_archive_once(safe, self.root)

    def test_content_safe_split_quarantines_train_only_and_keeps_groups(self):
        import shutil
        shutil.copyfile(self.root / "train-images/1.png", self.root / "train-images/2.png")
        shutil.copyfile(self.root / "train-images/3.png", self.root / "public-test-images/100.png")
        train, test = [read_annotations(self.root / f"{name}_data.json") for name in ("train", "test")]
        test_before = (self.root / "test_data.json").read_bytes()
        audit = {"source_sha256": self.split["source_sha256"], "image_content_sha256": {}}
        for name, source, directory in (("official_train", train, "train-images"),
                                         ("official_test", test, "public-test-images")):
            audit["image_content_sha256"][name] = {
                image["filename"]: sha256_file(self.root / directory / image["filename"])
                for image in source["images"]}
        safe_path = self.root / "splits/safe.json"
        safe = create_content_safe_split(self.root, safe_path, audit, validation_fraction=.25)
        self.assertEqual(safe["excluded_train_ids_due_to_test_overlap"], [3])
        self.assertEqual(safe["excluded_train_images_due_to_test_overlap"][0]["filename"], "3.png")
        self.assertEqual(safe["test_image_ids"], [100, 101])
        self.assertEqual((self.root / "test_data.json").read_bytes(), test_before)
        self.assertEqual(safe["duplicate_groups"][0]["image_ids"], [1, 2])
        group = safe["duplicate_groups"][0]
        self.assertTrue({1, 2} <= set(safe[f"{group['assigned_split']}_image_ids"]))
        self.assertEqual(safe, create_content_safe_split(self.root, safe_path, audit, validation_fraction=.25))
        self.assertEqual(load_split(self.root, safe_path), safe)
        # Intentionally divide an image-content group while keeping ID counts valid.
        broken = deepcopy(safe)
        old = group["assigned_split"]
        new = "val" if old == "train" else "train"
        broken[f"{old}_image_ids"].remove(1)
        broken[f"{new}_image_ids"].append(1)
        for name in (old, new):
            n = len(broken[f"{name}_image_ids"])
            broken["counts"][name] = {"images": n, "annotations": n * 2}
        with self.assertRaisesRegex(ValueError, "Exact image bytes overlap"):
            validate_partition(train, test, broken)

    def test_image_audit_detects_corrupt_and_missing_files(self):
        (self.root / "train-images/1.png").write_bytes(b"broken image")
        (self.root / "train-images/2.png").unlink()
        result, _ = image_audit(read_annotations(self.root / "train_data.json"), self.root / "train-images")
        self.assertEqual(result["readable_images"], 6)
        self.assertEqual(result["missing"], ["2.png"])
        self.assertEqual(result["corrupt"][0]["filename"], "1.png")


class CollatorTests(unittest.TestCase):
    def test_right_padding_and_mask_semantics(self):
        samples = [{"images": torch.ones(3, 32, 32), "captions": torch.tensor(ids),
                    "image_id": index, "filename": f"{index}.png", "caption_text": "example",
                    "reference_captions": ["example"]}
                   for index, ids in enumerate(([1, 4, 5, 2], [1, 2]))]
        batch = CaptionCollator(0)(samples)
        self.assertEqual(batch["captions"].tolist(), [[1, 4, 5, 2], [1, 2, 0, 0]])
        self.assertEqual(batch["padding_mask"].tolist(), [[False] * 4, [False, False, True, True]])
        self.assertEqual(batch["lengths"].tolist(), [4, 2])


if __name__ == "__main__":
    unittest.main()
