"""Inspect/extract raw assets, reproduce the fixed split/vocabs, and audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import build_training_vocabulary, create_content_safe_split, create_split
from src.data.audit import extract_archive_once, run_audit
from src.data.tokenizer import CAPTION_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--content-safe", action="store_true",
                        help="Also create a grouped content-safe split; preserve the requested ID-only split")
    args = parser.parse_args()
    split_file = args.split_file or args.data_dir / "splits" / "ktvic_seed42.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction = []
    if not args.skip_extract:
        # These are the two archive names discovered in the actual workspace.
        for filename in ("train-images.zip", "public-test-images.zip"):
            path = args.data_dir / filename
            if path.is_file():
                print(f"Checking/extracting {path.name}...", flush=True)
                extraction.append(extract_archive_once(path, args.data_dir))
    split = None
    vocabularies = None
    partition_error = None
    try:
        split = create_split(args.data_dir, split_file, seed=42, validation_fraction=.15)
        print(f"Fixed split: {split['counts']}", flush=True)
        vocabularies = {}
        for field in CAPTION_FIELDS:
            vocabulary = build_training_vocabulary(args.data_dir, split, field, min_frequency=2)
            vocabulary.save(args.output_dir / f"vocab_{field}_seed42.json")
            vocabularies[field] = vocabulary
    except ValueError as exc:
        partition_error = str(exc)
        print(f"Split blocked, continuing with raw audit: {exc}", flush=True)
    report = run_audit(args.data_dir, args.output_dir, split, vocabularies)
    if args.content_safe and not partition_error:
        # Preserve the requested split's audit, including its content leakage failure.
        (args.output_dir / "data_audit_id_only.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        safe_path = args.data_dir / "splits" / "ktvic_seed42_content_safe.json"
        safe = create_content_safe_split(args.data_dir, safe_path, report)
        print(f"Content-safe split: {safe['counts']}; quarantined IDs {safe['excluded_train_image_ids']}", flush=True)
        safe_vocabs = {}
        for field in CAPTION_FIELDS:
            vocabulary = build_training_vocabulary(args.data_dir, safe, field, min_frequency=2)
            vocabulary.save(args.output_dir / f"vocab_{field}_seed42.json")
            safe_vocabs[field] = vocabulary
        report = run_audit(args.data_dir, args.output_dir, safe, safe_vocabs, verified_image_cache=report)
    (args.output_dir / "extraction_report.json").write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit {report['status']}: {args.output_dir / 'data_audit.json'}", flush=True)
    if partition_error or report["blockers"]:
        raise SystemExit(partition_error or "; ".join(report["blockers"]))


if __name__ == "__main__":
    main()
