#!/usr/bin/env python3
"""
Create competition submission ZIP.

Packages solution.py + ONNX models + weights into submission.zip.

Usage:
    python scripts/create_submission.py
    python scripts/create_submission.py --output submissions/v3.zip
"""

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import current_git_commit, sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description='Create submission ZIP')
    parser.add_argument(
        '--output',
        type=str,
        default='submissions/submission.zip',
        help='Output zip path',
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/submission',
        help='Curated directory whose artifacts get packaged (paths preserved relative to repo root)',
    )
    parser.add_argument(
        '--allow-fallback-package',
        action='store_true',
        help='Allow packaging without model artifacts for explicit smoke tests',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = (ROOT / models_dir).resolve()

    # Required files
    solution_py = ROOT / 'solution.py'
    if not solution_py.exists():
        print("ERROR: solution.py not found at project root!")
        sys.exit(1)

    artifact_suffixes = {'.onnx', '.pt', '.pth', '.joblib', '.pkl', '.npy', '.json', '.yaml', '.yml'}
    artifact_files = [
        p for p in sorted(models_dir.rglob('*'))
        if p.is_file() and p.suffix.lower() in artifact_suffixes
    ]
    model_files = [
        p for p in artifact_files
        if p.suffix.lower() in {'.onnx', '.pt', '.pth', '.joblib', '.pkl'}
    ]
    if not model_files and not args.allow_fallback_package:
        print(
            f"ERROR: No trained model artifacts found in {models_dir}/. "
            "Refusing to create a silent fallback submission."
        )
        sys.exit(1)

    artifact_manifest = []
    for artifact_path in artifact_files:
        arcname = artifact_path.relative_to(ROOT)
        artifact_manifest.append(
            {
                "zip_path": str(arcname).replace('\\', '/'),
                "source_path": str(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            }
        )
    submission_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "solution_py_sha256": sha256_file(solution_py),
        "fallback_allowed": bool(args.allow_fallback_package),
        "model_artifact_count": len(model_files),
        "artifact_count": len(artifact_files),
        "artifacts": artifact_manifest,
    }

    # Build ZIP
    with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        # solution.py at root
        zf.write(str(solution_py), 'solution.py')
        zf.writestr(
            'submission_manifest.json',
            json.dumps(submission_manifest, indent=2, sort_keys=True),
        )

        for artifact_path in artifact_files:
            arcname = artifact_path.relative_to(ROOT)
            zf.write(str(artifact_path), str(arcname).replace('\\', '/'))

        # PyTorch checkpoints need source modules at inference time.
        if any(p.suffix.lower() in {'.pt', '.pth'} for p in model_files):
            for src_path in sorted((ROOT / 'src').rglob('*.py')):
                arcname = src_path.relative_to(ROOT)
                zf.write(str(src_path), str(arcname).replace('\\', '/'))

        # Optional README
        readme = ROOT / 'README.md'
        if readme.exists():
            zf.write(str(readme), 'README.md')

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Submission created: {output_path} ({size_mb:.1f} MB)")
    print("Contents:")
    with zipfile.ZipFile(str(output_path), 'r') as zf:
        names = zf.namelist()
        if 'solution.py' not in names:
            print("ERROR: solution.py is not at zip root")
            sys.exit(1)
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
