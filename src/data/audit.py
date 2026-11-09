"""Reproducible raw-text, archive, and image integrity audit (no raw edits)."""

from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import statistics
import unicodedata
import zipfile
import zlib

from PIL import Image

from src.data.annotations import read_annotations, source_hashes, split_fingerprint
from src.data.tokenizer import CAPTION_FIELDS, tokenize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def summary(values: list[int | float]) -> dict:
    """Linear-interpolated percentiles, matching NumPy's default definition."""
    if not values:
        return {"count": 0}
    values = sorted(values)

    def percentile(q: float) -> float:
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p90": percentile(.90), "p95": percentile(.95), "p99": percentile(.99)}


def duplicate_values(values: list) -> list[dict]:
    return [{"value": value, "count": count}
            for value, count in Counter(values).items() if count > 1]


def structure_audit(source: dict) -> dict:
    images, annotations = source["images"], source["annotations"]
    ids = [image["id"] for image in images]
    known = set(ids)
    counts = Counter(a["image_id"] for a in annotations)
    distribution = Counter(counts[image_id] for image_id in known)
    return {"images": len(images), "annotations": len(annotations),
            "captions_per_image": {str(n): count for n, count in sorted(distribution.items())},
            "duplicate_image_ids": duplicate_values(ids),
            "duplicate_filenames": duplicate_values([i["filename"] for i in images]),
            "duplicate_annotation_ids": duplicate_values([a["id"] for a in annotations]),
            "orphan_annotation_ids": [a["id"] for a in annotations if a["image_id"] not in known],
            "images_without_captions": sorted(known - set(counts)),
            "invalid_caption_fields": [
                {"annotation_id": a["id"], "field": field}
                for a in annotations for field in CAPTION_FIELDS
                if not isinstance(a.get(field), str) or not a[field].strip()]}


def text_audit(source: dict, field: str) -> dict:
    annotations = [a for a in source["annotations"] if isinstance(a.get(field), str)]
    captions = [a[field] for a in annotations]
    tokens = [tokenize(caption) for caption in captions]
    counts = Counter(token for caption in tokens for token in caption)
    occurrences = defaultdict(list)
    normalized = Counter()
    for annotation in annotations:
        occurrences[annotation[field]].append(annotation)
        normalized[" ".join(tokenize(annotation[field]))] += 1
    duplicate_captions = []
    for text, rows in occurrences.items():
        if len(rows) > 1:
            duplicate_captions.append({"text": text, "count": len(rows),
                                       "image_ids": sorted({r["image_id"] for r in rows}),
                                       "annotation_ids": [r["id"] for r in rows]})
    duplicate_captions.sort(key=lambda row: (-row["count"], row["text"]))
    rare = [{"token": token, "count": count}
            for token, count in sorted(counts.items(), key=lambda row: (row[1], row[0])) if count <= 2]
    unusual = []
    for token, count in sorted(counts.items()):
        reasons = []
        if any(char.isdigit() for char in token):
            reasons.append("contains_digit")
        if any(not (char.isalpha() or char == "_") for char in token):
            reasons.append("punctuation_or_symbol")
        if "\ufffd" in token:
            reasons.append("unicode_replacement_character")
        if len(token) > 35:
            reasons.append("long_token")
        if reasons:
            unusual.append({"token": token, "count": count, "reasons": reasons})
    # Heuristic review hints, NOT spelling corrections or evidence of an error.
    common = sorted(token for token, count in counts.items() if count >= 20)
    hints = []
    for row in rare:
        token = row["token"]
        if len(token) >= 5 and all(char.isalpha() or char == "_" for char in token):
            matches = difflib.get_close_matches(token, common, n=1, cutoff=.88)
            if matches:
                hints.append({**row, "similar_common_token": matches[0],
                              "common_count": counts[matches[0]]})
    return {"caption_count": len(captions), "length_tokens": summary([len(t) for t in tokens]),
            "vocabulary_size": len(counts), "total_tokens": sum(counts.values()),
            "token_frequency": dict(sorted(counts.items(), key=lambda row: (-row[1], row[0]))),
            "frequency_histogram": dict(sorted(Counter(counts.values()).items())),
            "rare_tokens_frequency_at_most_2": rare,
            "unusual_tokens": unusual,
            "possible_typo_hints": hints,
            "typo_hint_method": "rare token <=2; difflib similarity >=0.88 to token >=20; manual review only",
            "non_nfc_caption_count": sum(text != unicodedata.normalize("NFC", text) for text in captions),
            "duplicate_caption_groups": len(duplicate_captions),
            "duplicate_caption_excess_rows": sum(row["count"] - 1 for row in duplicate_captions),
            "duplicate_captions": duplicate_captions,
            "normalized_duplicate_groups": sum(count > 1 for count in normalized.values())}


def oov_audit(captions: list[str], known_tokens: set[str]) -> dict:
    counts = Counter(token for text in captions for token in tokenize(text))
    oov = {token: count for token, count in counts.items() if token not in known_tokens}
    total = sum(counts.values())
    return {"vocabulary_size": len(counts), "total_tokens": total,
            "oov_types": len(oov), "oov_tokens": sum(oov.values()),
            "oov_type_rate": len(oov) / len(counts) if counts else 0,
            "oov_token_rate": sum(oov.values()) / total if total else 0,
            "oov_token_frequency": dict(sorted(oov.items(), key=lambda row: (-row[1], row[0])))}


def inspect_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        files = [m for m in members if not m.is_dir()]
        unsafe = []
        for member in members:
            # Backslashes also act as separators on Windows.
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if (name.is_absolute() or ".." in name.parts or ":" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)):
                unsafe.append(member.filename)
        return {"file": path.name, "members": len(members), "files": len(files),
                "uncompressed_bytes": sum(m.file_size for m in files),
                "root_entries": sorted({PurePosixPath(m.filename).parts[0] for m in members}),
                "extension_counts": dict(Counter(PurePosixPath(m.filename).suffix.lower() for m in files)),
                "duplicate_members": duplicate_values([m.filename for m in members]),
                "unsafe_members": unsafe, "member_sample": [m.filename for m in members[:8]]}


def extract_archive_once(archive_path: Path, data_dir: Path) -> dict:
    """Validate every path first; skip existing identical files; never overwrite."""
    inspection = inspect_archive(archive_path)
    if inspection["unsafe_members"] or inspection["duplicate_members"]:
        raise ValueError(f"Unsafe/duplicate members in {archive_path}")
    root = data_dir.resolve()
    written = skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        # Validate all resolved paths before writing any file (also handles symlinks).
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path escapes data directory: {member.filename}")
        for member in archive.infolist():
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ValueError(f"Refusing to replace a different existing file: {target}")
                checksum = 0
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum = zlib.crc32(chunk, checksum)
                if checksum & 0xffffffff != member.CRC:
                    raise ValueError(f"Existing file CRC differs from archive: {target}")
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            written += 1
    return {"archive": archive_path.name, "extracted": written,
            "existing_identical": skipped, "integrity": "CRC checked during extraction or comparison"}


def image_audit(source: dict, image_dir: Path) -> tuple[dict, dict[str, list[str]]]:
    """Verify headers and decode pixels once; read dimensions without OCR/resizing."""
    missing, corrupt, unsafe, unexpected_extensions = [], [], [], []
    widths, heights, ratios = [], [], []
    extensions, formats, modes = Counter(), Counter(), Counter()
    content_hashes = defaultdict(list)
    expected = {image["filename"] for image in source["images"]}
    for image in source["images"]:
        filename = image["filename"]
        path = (image_dir / filename).resolve()
        if not path.is_relative_to(image_dir.resolve()):
            unsafe.append(filename)
            continue
        if not path.is_file():
            missing.append(filename)
            continue
        extension = path.suffix.lower()
        extensions[extension] += 1
        if extension not in IMAGE_EXTENSIONS:
            unexpected_extensions.append(filename)
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            content_hashes[digest.hexdigest()].append(filename)
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                opened.load()
                width, height = opened.size
                widths.append(width)
                heights.append(height)
                ratios.append(width / height)
                formats[opened.format] += 1
                modes[opened.mode] += 1
        except Exception as exc:
            corrupt.append({"filename": filename, "error": f"{type(exc).__name__}: {exc}"})
    actual = {p.relative_to(image_dir).as_posix() for p in image_dir.rglob("*")
              if p.is_file()} if image_dir.exists() else set()
    report = {"directory": image_dir.name, "expected_images": len(source["images"]),
              "readable_images": len(widths), "missing": missing, "corrupt": corrupt,
              "unsafe_filenames": unsafe, "extra_files": sorted(actual - expected),
              "unexpected_extensions": unexpected_extensions,
              "extension_counts": dict(extensions), "decoded_format_counts": dict(formats),
              "color_modes": dict(modes), "width": summary(widths), "height": summary(heights),
              "aspect_ratio_width_over_height": summary(ratios),
              "exact_byte_duplicate_groups": [names for names in content_hashes.values() if len(names) > 1]}
    return report, dict(content_hashes)


def run_audit(data_dir: Path, output_dir: Path, split: dict | None = None,
              vocabularies: dict | None = None, verified_image_cache: dict | None = None) -> dict:
    train = read_annotations(data_dir / "train_data.json")
    test = read_annotations(data_dir / "test_data.json")
    report = {"schema_version": 1, "dataset": "KTVIC", "source_sha256": source_hashes(data_dir),
              "tokenizer": "NFC + lowercase + whitespace split; underscore preserved; no special tokens in lengths",
              "notes": ["Official test used only for descriptive audit, never for fitting vocabulary or choosing settings.",
                        "Duplicate caption text and heuristic typo candidates do not prove duplicate images or spelling errors.",
                        "Image hashes detect exact encoded-file duplicates, not perceptual near duplicates."],
              "archives": [inspect_archive(p) for p in sorted(data_dir.glob("*.zip"))],
              "structure": {"official_train": structure_audit(train), "official_test": structure_audit(test)},
              "text": {}, "images": {}}
    report["overlap"] = {
        "image_ids": sorted({i["id"] for i in train["images"]} & {i["id"] for i in test["images"]}),
        "filenames": sorted({i["filename"] for i in train["images"]} & {i["filename"] for i in test["images"]})}
    for field in CAPTION_FIELDS:
        train_text, test_text = text_audit(train, field), text_audit(test, field)
        train_captions = [a[field] for a in train["annotations"] if isinstance(a.get(field), str)]
        test_captions = [a[field] for a in test["annotations"] if isinstance(a.get(field), str)]
        shared = sorted(set(train_captions) & set(test_captions))
        field_report = {"official_train": train_text, "official_test": test_text,
                        "test_oov_vs_official_train": oov_audit(test_captions, set(train_text["token_frequency"])),
                        "train_test_exact_shared_caption_count": len(shared),
                        "train_test_exact_shared_captions": shared}
        if split is not None:
            train_ids = set(split["train_image_ids"])
            val_ids = set(split["val_image_ids"])
            fitting = [a[field] for a in train["annotations"] if a["image_id"] in train_ids]
            validation = [a[field] for a in train["annotations"] if a["image_id"] in val_ids]
            training_tokens = {token for text in fitting for token in tokenize(text)}
            field_report["val_oov_vs_derived_train"] = oov_audit(validation, training_tokens)
            field_report["test_oov_vs_derived_train"] = oov_audit(test_captions, training_tokens)
            if vocabularies is not None:
                vocabulary = vocabularies[field]
                known = set(vocabulary.token_to_id)
                field_report["fitted_vocabulary"] = {
                    "size_including_4_special_tokens": len(vocabulary),
                    "min_frequency": vocabulary.metadata["min_frequency"], "fit_split": "train",
                    "train_encoding_oov": oov_audit(fitting, known),
                    "val_encoding_oov": oov_audit(validation, known),
                    "test_encoding_oov": oov_audit(test_captions, known),
                    "train_captions_exceeding_default_64_with_bos_eos": sum(len(tokenize(t)) + 2 > 64 for t in fitting)}
        report["text"][field] = field_report
    if verified_image_cache is None:
        print("Text audit complete; verifying every referenced image...", flush=True)
        train_images, train_hashes = image_audit(train, data_dir / "train-images")
        print(f"Official train images readable: {train_images['readable_images']}", flush=True)
        test_images, test_hashes = image_audit(test, data_dir / "public-test-images")
    else:
        # Reuse only within this preparation invocation after the full pixel audit.
        train_images = verified_image_cache["images"]["official_train"]
        test_images = verified_image_cache["images"]["official_test"]
        hashes = {}
        for name in ("official_train", "official_test"):
            by_digest = defaultdict(list)
            for filename, digest in verified_image_cache["image_content_sha256"][name].items():
                by_digest[digest].append(filename)
            hashes[name] = dict(by_digest)
        train_hashes, test_hashes = hashes["official_train"], hashes["official_test"]
    report["images"] = {"official_train": train_images, "official_test": test_images}
    report["image_content_sha256"] = {
        "official_train": {filename: digest for digest, names in train_hashes.items() for filename in names},
        "official_test": {filename: digest for digest, names in test_hashes.items() for filename in names}}
    report["overlap"]["exact_image_bytes"] = [
        {"sha256": digest, "train_filenames": train_hashes[digest], "test_filenames": test_hashes[digest]}
        for digest in sorted(set(train_hashes) & set(test_hashes))]
    report["split"] = ({"seed": split["seed"], "counts": split["counts"],
                        "fingerprint": split_fingerprint(split),
                        "partition_policy": split.get("partition_policy", "image_id_only"),
                        "excluded_train_image_ids": split.get("excluded_train_image_ids", []),
                        "coverage": split.get("coverage", "train + val == official_train; official_test unchanged"),
                        "assertions": "PASS: disjoint IDs/filenames, exact declared source coverage, all retained references preserved"}
                       if split is not None else None)
    report["active_partition_content_overlap"] = {}
    if split is not None:
        lookup = {name: {i["id"]: i["filename"] for i in source["images"]}
                  for name, source in (("official_train", train), ("official_test", test))}
        content_sets = {}
        for name in ("train", "val", "test"):
            source_name = "official_test" if name == "test" else "official_train"
            by_filename = report["image_content_sha256"][source_name]
            content_sets[name] = {by_filename[lookup[source_name][i]] for i in split[f"{name}_image_ids"]
                                  if lookup[source_name][i] in by_filename}
        report["active_partition_content_overlap"] = {
            f"{left}_{right}": sorted(content_sets[left] & content_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
    blockers = []
    for name, result in report["structure"].items():
        for key in ("duplicate_image_ids", "duplicate_filenames", "duplicate_annotation_ids",
                    "orphan_annotation_ids", "images_without_captions", "invalid_caption_fields"):
            if result[key]:
                blockers.append(f"{name}: {key}")
    for name, result in report["images"].items():
        for key in ("missing", "corrupt", "unsafe_filenames"):
            if result[key]:
                blockers.append(f"{name}: {key}")
    for key in ("image_ids", "filenames"):
        if report["overlap"][key]:
            blockers.append(f"Official train/test overlap: {key}")
    for pair, overlap in report["active_partition_content_overlap"].items():
        if overlap:
            blockers.append(f"Active partitions overlap in exact image bytes: {pair}")
    if split is None and report["overlap"]["exact_image_bytes"]:
        blockers.append("Official train/test overlap: exact_image_bytes")
    report["raw_data_warnings"] = []
    if report["overlap"]["exact_image_bytes"]:
        report["raw_data_warnings"].append("Official train/test share exact image bytes; see overlap.exact_image_bytes")
    if train_images["exact_byte_duplicate_groups"]:
        report["raw_data_warnings"].append("Official train contains byte-identical images under different IDs/filenames")
    report["blockers"] = blockers
    report["status"] = "PASS" if not blockers else "FAIL"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (output_dir / "data_audit.md").write_text(audit_markdown(report), encoding="utf-8")
    return report


def audit_markdown(report: dict) -> str:
    lines = ["# KTVIC data audit", "", f"Status: **{report['status']}**", "",
             "Reproduce: `python scripts/prepare_data.py` (no network).",
             "Raw JSON is unchanged. Full frequencies, duplicate groups, OOV lists, hashes,",
             "image integrity, and heuristic review hints are in `data_audit.json`.", "",
             "## Structure", "", "| Source | Images | Captions | Captions per image |", "|---|---:|---:|---|"]
    for name, result in report["structure"].items():
        lines.append(f"| {name} | {result['images']} | {result['annotations']} | {result['captions_per_image']} |")
    lines += ["", f"ID overlap: {len(report['overlap']['image_ids'])}; filename overlap: "
              f"{len(report['overlap']['filenames'])}; exact image byte overlap: {len(report['overlap']['exact_image_bytes'])}.",
              "", "## Text", "", "Lengths use NFC/lowercase/whitespace; underscores remain in segmented tokens.",
              "No BOS/EOS tokens are included. Percentiles use linear interpolation.", "",
              "| Field / source | Mean | Median | Min / max | P90 / P95 / P99 | Vocab | Duplicate groups |",
              "|---|---:|---:|---|---|---:|---:|"]
    for field, field_report in report["text"].items():
        for source in ("official_train", "official_test"):
            result = field_report[source]
            length = result["length_tokens"]
            if not length["count"]:
                continue
            lines.append(f"| {field} / {source} | {length['mean']:.2f} | {length['median']} | "
                         f"{length['min']} / {length['max']} | {length['p90']:.1f} / {length['p95']:.1f} / "
                         f"{length['p99']:.1f} | {result['vocabulary_size']} | {result['duplicate_caption_groups']} |")
    for field, result in report["text"].items():
        oov = result["test_oov_vs_official_train"]
        lines += ["", f"### {field}", "",
                  f"Test OOV vs official train: {oov['oov_types']} types, {oov['oov_tokens']} tokens "
                  f"({oov['oov_token_rate']:.3%}). Shared exact train/test caption strings: "
                  f"{result['train_test_exact_shared_caption_count']}."]
        if "fitted_vocabulary" in result:
            vocab = result["fitted_vocabulary"]
            lines.append(f"Vocabulary fitted on derived train only: {vocab['size_including_4_special_tokens']} IDs "
                         f"including 4 special tokens; min_frequency={vocab['min_frequency']}.")
            lines.append(f"Actual encoding OOV: val {vocab['val_encoding_oov']['oov_token_rate']:.3%}; "
                         f"test {vocab['test_encoding_oov']['oov_token_rate']:.3%}.")
        rare = result["official_train"]["rare_tokens_frequency_at_most_2"]
        unusual = result["official_train"]["unusual_tokens"]
        hints = result["official_train"]["possible_typo_hints"]
        lines += [f"Rare train tokens (frequency <=2): {len(rare)}; unusual tokens: {len(unusual)}; "
                  f"heuristic similarity hints: {len(hints)}.", "",
                  "Possible typo hints (manual review only; data was not corrected):"]
        lines += [f"- `{h['token']}` ({h['count']}) resembles `{h['similar_common_token']}` "
                  f"({h['common_count']})." for h in hints[:12]] or ["- None."]
    lines += ["", "## Images", "", "| Source | Readable | Missing | Corrupt | Formats | Width | Height | Aspect ratio |",
              "|---|---:|---:|---:|---|---|---|---|"]
    for name, result in report["images"].items():
        def bounds(key):
            info = result[key]
            return f"{info['min']:.2f}–{info['max']:.2f}; mean {info['mean']:.2f}" if info["count"] else "n/a"
        lines.append(f"| {name} | {result['readable_images']} | {len(result['missing'])} | {len(result['corrupt'])} | "
                     f"{result['extension_counts']} | {bounds('width')} | {bounds('height')} | "
                     f"{bounds('aspect_ratio_width_over_height')} |")
    if report["split"]:
        lines += ["", "## Fixed split", "", f"Seed: {report['split']['seed']}. {report['split']['assertions']}.", ""]
        for name, count in report["split"]["counts"].items():
            lines.append(f"- {name}: {count['images']} images, {count['annotations']} captions.")
        lines += ["", f"Policy: `{report['split']['partition_policy']}`.",
                  f"Coverage: {report['split']['coverage']}.",
                  f"Quarantined train IDs: {report['split']['excluded_train_image_ids']}.",
                  f"Active byte-overlap counts: { {pair: len(values) for pair, values in report['active_partition_content_overlap'].items()} }."]
    lines += ["", "## Raw data identity findings", ""]
    for group in report["overlap"]["exact_image_bytes"]:
        lines.append(f"- Train {group['train_filenames']} = test {group['test_filenames']} (same SHA256).")
    for group in report["images"]["official_train"]["exact_byte_duplicate_groups"]:
        lines.append(f"- Duplicate official train files: {group}.")
    lines += ["", "## Limits and blockers", "",
              "Official test statistics are descriptive; they do not set vocabulary or hyperparameters.",
              "Exact byte hashing does not detect re-encoded or visually similar images.",
              "Repeated generic captions alone are not evidence of image leakage.", ""]
    lines += [f"- {blocker}" for blocker in report["blockers"]] or ["- No data integrity blockers found."]
    return "\n".join(lines) + "\n"
