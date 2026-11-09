"""Read KTVIC annotations and enforce an immutable image-level partition."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ImageId = int | str


def read_annotations(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(data.get(k), list)
                                             for k in ("images", "annotations")):
        raise ValueError(f"Invalid COCO-like annotation structure: {path}")
    return data


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(data_dir: Path) -> dict[str, str]:
    return {name: sha256_file(data_dir / name)
            for name in ("train_data.json", "test_data.json")}


def split_fingerprint(split: dict) -> str:
    """Canonical JSON fingerprint independent of formatting or key order."""
    return hashlib.sha256(json.dumps(split, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_partition(official_train: dict, official_test: dict, split: dict) -> None:
    """Fail on leaked, missing, duplicated IDs, filenames, or lost references."""
    for name, source in (("official_train", official_train), ("official_test", official_test)):
        ids = [image["id"] for image in source["images"]]
        names = [image["filename"] for image in source["images"]]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError(f"Duplicate IDs/filenames in {name}")
        known = set(ids)
        counts = Counter(a["image_id"] for a in source["annotations"])
        if set(counts) != known:
            raise ValueError(f"Orphan annotations or uncaptioned images in {name}")
        for annotation in source["annotations"]:
            for field in ("caption", "segment_caption"):
                if not isinstance(annotation.get(field), str) or not annotation[field].strip():
                    raise ValueError(f"Missing/empty {field} in {name}")
    official_ids = {image["id"] for image in official_train["images"]}
    test_ids = {image["id"] for image in official_test["images"]}
    parts = {}
    for key in ("train", "val", "test"):
        ids = split[f"{key}_image_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {key} split")
        parts[key] = set(ids)
    if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]:
        raise ValueError("Image IDs overlap across splits")
    excluded_list = split.get("excluded_train_image_ids", [])
    excluded = set(excluded_list)
    if (len(excluded) != len(excluded_list) or not excluded <= official_ids
            or excluded & (parts["train"] | parts["val"])):
        raise ValueError("Invalid quarantine IDs")
    if excluded and split.get("partition_policy") != "group_exact_sha256_exclude_train_test_duplicates":
        raise ValueError("Quarantine requires an explicit content-safe partition policy")
    if parts["train"] | parts["val"] | excluded != official_ids or parts["test"] != test_ids:
        raise ValueError("Splits do not exactly cover the official sources")
    filenames = {image["filename"] for image in official_train["images"]}
    if filenames & {image["filename"] for image in official_test["images"]}:
        raise ValueError("Official train/test filenames overlap")
    original_counts = Counter(a["image_id"] for a in official_train["annotations"])
    for key in ("train", "val", "test"):
        source = official_test if key == "test" else official_train
        selected = [a for a in source["annotations"] if a["image_id"] in parts[key]]
        expected = Counter(a["image_id"] for a in source["annotations"])
        actual = Counter(a["image_id"] for a in selected)
        if any(actual[image_id] != expected[image_id] for image_id in parts[key]):
            raise ValueError(f"Lost reference captions in {key}")
        if "counts" in split:
            if split["counts"][key] != {"images": len(parts[key]), "annotations": len(selected)}:
                raise ValueError(f"Stale count metadata in {key}")
    if sum(original_counts.values()) != sum(
        original_counts[i] for i in parts["train"] | parts["val"] | excluded
    ):
        raise ValueError("Official train references were lost")
    if "content_sha256" in split:
        hashes = {source: {row["image_id"]: row["sha256"] for row in rows}
                  for source, rows in split["content_sha256"].items()}
        if set(hashes["official_train"]) != official_ids or set(hashes["official_test"]) != test_ids:
            raise ValueError("Content hash manifest does not cover the official sources")
        test_hashes = set(hashes["official_test"].values())
        expected_excluded = {i for i, digest in hashes["official_train"].items() if digest in test_hashes}
        if excluded != expected_excluded:
            raise ValueError("Quarantine does not exactly match train images also present in test")
        content_sets = {name: {hashes["official_test" if name == "test" else "official_train"][i]
                              for i in parts[name]} for name in ("train", "val", "test")}
        if (content_sets["train"] & content_sets["val"] or content_sets["train"] & content_sets["test"]
                or content_sets["val"] & content_sets["test"]):
            raise ValueError("Exact image bytes overlap across active partitions")
        if split.get("counts", {}).get("excluded") != {
            "images": len(excluded), "annotations": sum(original_counts[i] for i in excluded)
        }:
            raise ValueError("Stale quarantine count metadata")
        if "excluded_train_ids_due_to_test_overlap" in split:
            if split["excluded_train_ids_due_to_test_overlap"] != sorted(excluded):
                raise ValueError("Exclusion metadata does not match quarantined IDs")
        for group in split.get("duplicate_groups", []):
            memberships = {name for name in ("train", "val", "excluded")
                           if set(group["image_ids"]) & (excluded if name == "excluded" else parts[name])}
            if len(memberships) != 1 or group["assigned_split"] not in memberships:
                raise ValueError("A duplicate group was divided or has stale metadata")
        if "official_test_integrity" in split:
            actual = hashlib.sha256(json.dumps(official_test["annotations"], sort_keys=True,
                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            if split["official_test_integrity"]["annotations_sha256"] != actual:
                raise ValueError("Official test annotations were changed")


def create_split(data_dir: Path, destination: Path, seed: int = 42,
                 validation_fraction: float = 0.15) -> dict:
    """Shuffle sorted image IDs with local Python RNG; reserve round(n * .15)."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    train = read_annotations(data_dir / "train_data.json")
    test = read_annotations(data_dir / "test_data.json")
    ids = sorted(image["id"] for image in train["images"])
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * validation_fraction)
    if n_val == 0 or n_val == len(ids):
        raise ValueError("Need nonempty train and validation splits")
    split = {
        "schema_version": 1, "dataset": "KTVIC", "seed": seed,
        "validation_fraction": validation_fraction,
        "algorithm": "sorted_ids_then_python_random_Random_shuffle_first_round_n_fraction_for_val",
        "source_sha256": source_hashes(data_dir),
        "train_image_ids": sorted(ids[n_val:]), "val_image_ids": sorted(ids[:n_val]),
        "test_image_ids": sorted(image["id"] for image in test["images"]),
    }
    split["counts"] = {}
    for name in ("train", "val", "test"):
        selected = set(split[f"{name}_image_ids"])
        source = test if name == "test" else train
        split["counts"][name] = {
            "images": len(selected),
            "annotations": sum(a["image_id"] in selected for a in source["annotations"]),
        }
    validate_partition(train, test, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != split:
            raise ValueError(f"Refusing to overwrite a different fixed split: {destination}")
    else:
        destination.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def load_split(data_dir: Path | str, split_file: Path | str) -> dict:
    """Check provenance whenever a split is loaded, so changed raw JSON fails."""
    data_dir = Path(data_dir)
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if split["source_sha256"] != source_hashes(data_dir):
        raise ValueError("Raw annotation fingerprints differ from the fixed split")
    validate_partition(read_annotations(data_dir / "train_data.json"),
                       read_annotations(data_dir / "test_data.json"), split)
    return split


def load_train_val_split(data_dir: Path | str, split_file: Path | str) -> dict:
    """Validate the frozen split without opening official test annotations/images.

    Test identity/leakage checks use only the already frozen hash manifest. This
    does not re-audit the current official test files or claim to do so.
    """
    data_dir = Path(data_dir)
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if split.get("partition_policy") != "group_exact_sha256_exclude_train_test_duplicates":
        raise ValueError("Train/validation smoke requires the content-safe split")
    if sha256_file(data_dir / "train_data.json") != split["source_sha256"]["train_data.json"]:
        raise ValueError("Train annotations differ from the frozen split")
    source = read_annotations(data_dir / "train_data.json")
    ids = [image["id"] for image in source["images"]]
    names = [image["filename"] for image in source["images"]]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("Duplicate train IDs or filenames")
    references = Counter(a["image_id"] for a in source["annotations"])
    if set(references) != set(ids):
        raise ValueError("Orphan train annotations or uncaptioned images")
    for annotation in source["annotations"]:
        for field in ("caption", "segment_caption"):
            if not isinstance(annotation.get(field), str) or not annotation[field].strip():
                raise ValueError(f"Missing/empty {field} in train annotations")
    parts = {}
    for name in ("train", "val", "test", "excluded"):
        selected = split[f"{name}_image_ids"] if name != "excluded" else split["excluded_train_image_ids"]
        parts[name] = set(selected)
        if len(selected) != len(parts[name]):
            raise ValueError(f"Duplicate IDs in {name}")
    for index, name in enumerate(parts):
        if any(parts[name] & parts[other] for other in list(parts)[index + 1:]):
            raise ValueError("Split IDs overlap")
    if parts["train"] | parts["val"] | parts["excluded"] != set(ids):
        raise ValueError("Train/val/excluded do not cover official train")
    for name in ("train", "val", "excluded"):
        actual = {"images": len(parts[name]), "annotations": sum(references[i] for i in parts[name])}
        if split["counts"][name] != actual:
            raise ValueError(f"Stale {name} counts")
    manifests = split["content_sha256"]
    hashes = {}
    for source_name, expected_ids in (("official_train", set(ids)), ("official_test", parts["test"])):
        rows = manifests[source_name]
        hashes[source_name] = {row["image_id"]: row["sha256"] for row in rows}
        if len(rows) != len(hashes[source_name]) or set(hashes[source_name]) != expected_ids:
            raise ValueError(f"Incomplete/duplicated hash manifest: {source_name}")
        if any(len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
               for digest in hashes[source_name].values()):
            raise ValueError("Invalid SHA256 in frozen manifest")
    test_hashes = set(hashes["official_test"].values())
    expected_excluded = {i for i, digest in hashes["official_train"].items() if digest in test_hashes}
    if expected_excluded != parts["excluded"]:
        raise ValueError("Quarantine differs from frozen content manifest")
    train_hashes = {hashes["official_train"][i] for i in parts["train"]}
    val_hashes = {hashes["official_train"][i] for i in parts["val"]}
    if train_hashes & val_hashes or (train_hashes | val_hashes) & test_hashes:
        raise ValueError("Frozen manifest has active content leakage")
    if split["counts"]["test"]["images"] != len(parts["test"]):
        raise ValueError("Stale frozen test image count")
    for group in split.get("duplicate_groups", []):
        memberships = {name for name in ("train", "val", "excluded")
                       if set(group["image_ids"]) & parts[name]}
        if memberships != {group["assigned_split"]}:
            raise ValueError("Duplicate group was divided")
    return split


def default_split_file(data_dir: Path) -> Path:
    """Prefer the audited content-safe split, retaining the requested ID split."""
    safe = data_dir / "splits" / "ktvic_seed42_content_safe.json"
    return safe if safe.exists() else data_dir / "splits" / "ktvic_seed42.json"


def create_content_safe_split(data_dir: Path, destination: Path, audit: dict,
                              seed: int = 42, validation_fraction: float = .15) -> dict:
    """Group byte-identical images and quarantine train copies of test images.

    Raw sources remain intact. The exact coverage invariant becomes
    train + val + explicitly quarantined IDs == official train.
    Only image identity is used from test, never test captions or scores.
    """
    if audit["source_sha256"] != source_hashes(data_dir):
        raise ValueError("Audit source hashes are stale")
    train, test = [read_annotations(data_dir / f"{name}_data.json") for name in ("train", "test")]
    manifests = {}
    for name, source in (("official_train", train), ("official_test", test)):
        by_filename = audit["image_content_sha256"][name]
        manifests[name] = [{"image_id": image["id"], "sha256": by_filename[image["filename"]]}
                           for image in sorted(source["images"], key=lambda row: row["id"])]
    test_hashes = {row["sha256"] for row in manifests["official_test"]}
    excluded = sorted(row["image_id"] for row in manifests["official_train"] if row["sha256"] in test_hashes)
    grouped = {}
    for row in manifests["official_train"]:
        if row["sha256"] not in test_hashes:
            grouped.setdefault(row["sha256"], []).append(row["image_id"])
    groups = sorted(grouped.values(), key=lambda ids: ids[0])
    random.Random(seed).shuffle(groups)
    target = round(sum(len(ids) for ids in groups) * validation_fraction)
    train_ids, val_ids = [], []
    for ids in groups:
        if len(val_ids) + len(ids) <= target:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)
    if not train_ids or not val_ids:
        raise ValueError("Cannot form nonempty content-disjoint partitions at this fraction")
    split = {
        "schema_version": 1, "dataset": "KTVIC", "seed": seed,
        "validation_fraction": validation_fraction,
        "partition_policy": "group_exact_sha256_exclude_train_test_duplicates",
        "algorithm": "sort_groups_by_min_id_then_seeded_shuffle_then_greedy_fill_validation_image_target",
        "source_sha256": audit["source_sha256"], "content_sha256": manifests,
        "train_image_ids": sorted(train_ids), "val_image_ids": sorted(val_ids),
        "test_image_ids": sorted(image["id"] for image in test["images"]),
        "excluded_train_image_ids": excluded,
        "exclusion_reason": "Exact encoded-file SHA256 is also present in official test",
        "coverage": "train + val + excluded == official_train; official_test unchanged",
        "counts": {},
    }
    for name in ("train", "val", "test", "excluded"):
        ids = set(excluded if name == "excluded" else split[f"{name}_image_ids"])
        source = test if name == "test" else train
        split["counts"][name] = {"images": len(ids),
                                 "annotations": sum(a["image_id"] in ids for a in source["annotations"])}
    enrich_content_split_metadata(split, train, test)
    validate_partition(train, test, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != split:
            raise ValueError(f"Refusing to overwrite a different fixed split: {destination}")
    else:
        destination.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def enrich_content_split_metadata(split: dict, train: dict, test: dict) -> None:
    """Record every duplicate group and detailed quarantine provenance."""
    from collections import defaultdict

    train_images = {image["id"]: image for image in train["images"]}
    test_images = {image["id"]: image for image in test["images"]}
    groups = defaultdict(list)
    for row in split["content_sha256"]["official_train"]:
        groups[row["sha256"]].append(row["image_id"])
    test_groups = defaultdict(list)
    for row in split["content_sha256"]["official_test"]:
        test_groups[row["sha256"]].append(row["image_id"])
    memberships = {image_id: name for name in ("train", "val") for image_id in split[f"{name}_image_ids"]}
    memberships.update({image_id: "excluded" for image_id in split["excluded_train_image_ids"]})
    split["duplicate_groups"] = [
        {"duplicate_group": f"sha256:{digest}", "sha256": digest, "image_ids": sorted(ids),
         "filenames": [train_images[i]["filename"] for i in sorted(ids)],
         "assigned_split": memberships[ids[0]]}
        for digest, ids in sorted(groups.items()) if len(ids) > 1]
    split["excluded_train_ids_due_to_test_overlap"] = list(split["excluded_train_image_ids"])
    split["excluded_train_images_due_to_test_overlap"] = [
        {"image_id": image_id, "filename": train_images[image_id]["filename"], "sha256": digest,
         "reason": "Exact encoded-image SHA256 also occurs in official test; exclude only the train copy",
         "matching_test_images": [test_images[i] for i in sorted(test_groups[digest])]}
        for digest, ids in sorted(groups.items()) if digest in test_groups for image_id in sorted(ids)]
    split["official_test_integrity"] = {
        "images": len(test["images"]), "annotations": len(test["annotations"]),
        "raw_json_sha256": split["source_sha256"]["test_data.json"],
        "annotations_sha256": hashlib.sha256(json.dumps(test["annotations"], sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def build_training_vocabulary(data_dir: Path, split: dict, caption_field: str,
                              min_frequency: int = 2):
    from src.data.tokenizer import CAPTION_FIELDS, Vocabulary

    if caption_field not in CAPTION_FIELDS:
        raise ValueError(f"Unsupported caption field: {caption_field}")
    source = read_annotations(data_dir / "train_data.json")
    selected = set(split["train_image_ids"])
    captions = (a[caption_field] for a in source["annotations"] if a["image_id"] in selected)
    return Vocabulary.build(captions, min_frequency, metadata={
        "caption_field": caption_field, "fit_split": "train", "seed": split["seed"],
        "source_sha256": split["source_sha256"],
        "train_image_ids_sha256": hashlib.sha256(
            json.dumps(sorted(selected), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })
