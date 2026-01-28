#!/usr/bin/env python3
"""
add_kataforge_headers_safe.py

Safely adds KataForge license headers to .py and .nix files.
- Dry-run by default (use --apply to make changes)
- Creates backups
- Skips files that already contain a similar header
- Shows preview of changes

Usage:
    python3 add_kataforge_headers_safe.py /path/to/project [options]

Options:
    --apply       Actually modify files (default: dry-run only)
    --yes         Skip confirmation prompt when using --apply
    --force       Replace header even if a similar one is detected
    --verbose     Show more detailed output
"""

import os
import sys
import argparse
import re
from pathlib import Path
from difflib import unified_diff

# ──────────────────────────────────────────────────────────────────────────────
# Headers (customize if needed)
# ──────────────────────────────────────────────────────────────────────────────

HEADER_PYTHON = '''"""
KataForge - Adaptive Martial Arts Analysis System
Copyright © 2026 DeMoD LLC. All rights reserved.

This file is part of KataForge, released under the KataForge License
(based on Elastic License v2). See LICENSE in the project root for full terms.

SPDX-License-Identifier: Elastic-2.0

Description:
    [Brief module description – please edit]

Usage notes:
    - Private self-hosting, dojo use, and modifications are permitted.
    - Offering as a hosted/managed service to third parties is prohibited
      without explicit written permission from DeMoD LLC.
"""
'''

HEADER_NIX = '''# KataForge - Adaptive Martial Arts Analysis System
# Copyright © 2026 DeMoD LLC. All rights reserved.
#
# This file is part of KataForge, released under the KataForge License
# (based on Elastic License v2). See LICENSE in the project root for full terms.
#
# SPDX-License-Identifier: Elastic-2.0
#
# Description:
#   [Brief description of this Nix file – please edit]
#
# Usage notes:
#   - Private self-hosting and modifications are permitted per license terms.
#   - Hosted/managed service offerings are prohibited without permission.
'''

# ──────────────────────────────────────────────────────────────────────────────
# Header detection (more robust)
# ──────────────────────────────────────────────────────────────────────────────

def likely_has_header(content: str, suffix: str) -> bool:
    """Check if file already has a KataForge-style header"""
    key_phrases = [
        "Copyright © 2026 DeMoD LLC",
        "SPDX-License-Identifier: Elastic-2.0",
        "KataForge",
        "DeMoD LLC",
    ]
    return any(phrase in content for phrase in key_phrases)


def get_header_for_file(suffix: str) -> str | None:
    if suffix == ".py":
        return HEADER_PYTHON
    if suffix == ".nix":
        return HEADER_NIX
    return None


def preview_diff(old_content: str, new_content: str, path: Path) -> None:
    diff = unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (with header)",
    )
    print("".join(list(diff)[:12]))  # show first ~12 lines of diff
    if len(old_content.splitlines()) > 12:
        print("... (diff truncated)")


def add_header_if_needed(
    path: Path,
    dry_run: bool = True,
    force: bool = False,
    verbose: bool = False
) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    header = get_header_for_file(suffix)
    if not header:
        return False, "skipped (unsupported extension)"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"error reading file: {e}"

    if likely_has_header(content, suffix) and not force:
        return False, "skipped (header already present)"

    # Build new content
    if suffix == ".py":
        lines = content.splitlines(keepends=True)
        insert_pos = 0
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
        new_content = "".join(lines[:insert_pos]) + header.rstrip() + "\n\n" + "".join(lines[insert_pos:])
    else:  # .nix
        new_content = header.rstrip() + "\n\n" + content

    if dry_run:
        print(f"  Would update: {path}")
        if verbose:
            preview_diff(content, new_content, path)
        return False, "dry-run"

    # Real modification
    backup = path.with_name(f"{path.name}.bak")
    try:
        path.rename(backup)
        path.write_text(new_content, encoding="utf-8")
        return True, f"updated (backup: {backup.name})"
    except Exception as e:
        # Try to restore backup on failure
        if backup.exists():
            backup.rename(path)
        return False, f"failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="Safely add KataForge headers")
    parser.add_argument("path", help="Directory or file to process")
    parser.add_argument("--apply", action="store_true", help="Actually modify files")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation when --apply")
    parser.add_argument("--force", action="store_true", help="Replace existing headers")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diff previews")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if root.is_file():
        targets = [root]
    elif root.is_dir():
        targets = list(root.rglob("*.py")) + list(root.rglob("*.nix"))
    else:
        print(f"Error: {root} is not a valid file or directory", file=sys.stderr)
        return 1

    print(f"Found {len(targets)} candidate files (.py / .nix)")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}   Force: {args.force}")
    print("-" * 70)

    if args.apply and not args.yes:
        confirm = input("This will modify files. Continue? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return 0

    updated = 0
    skipped = 0

    for path in sorted(targets):
        changed, reason = add_header_if_needed(
            path, dry_run=not args.apply, force=args.force, verbose=args.verbose
        )
        print(f"{path}: {reason}")
        if changed:
            updated += 1
        else:
            skipped += 1

    print("\nSummary:")
    print(f"  Updated:     {updated}")
    print(f"  Skipped:     {skipped}")
    print(f"  Total files: {updated + skipped}")

    if not args.apply:
        print("\nNo files were modified. Use --apply to make changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
