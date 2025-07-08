"""
Copyright 2025 Zordi, Inc. All rights reserved.

Run the native *meshfix* binary on the three Galaxea STL meshes that remained
non-watertight after automatic trimesh repair:

    • arm_seg2.STL
    • arm_seg3.STL
    • galaxea_eoat_set.STL

A *_meshfix* copy is written next to each original file.  Pass --overwrite to
replace the originals (creates *.bak backups first).

Example usage from project root:

    # create *_meshfix.STL copies (safe default)
    python scripts/mesh/apply_meshfix_galaxea.py

    # overwrite originals (backs them up as *.bak)
    python scripts/mesh/apply_meshfix_galaxea.py --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FILES = [
    "arm_seg2.STL",
    "arm_seg3.STL",
    "galaxea_eoat_set.STL",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_meshfix(src: Path, dst: Path) -> bool:
    """Run meshfix on *src* and write to *dst*.  Return True on success."""
    print(f"[MESHFIX] {src.name:20s} → {dst.name}")
    # meshfix (Marco Attene) always writes <name>_fixed.off by default and does not
    # understand "-o" in some builds.  We therefore run it without extra args and
    # convert the resulting OFF to our desired STL path.
    fixed_off = src.with_name(src.stem + "_fixed.off")
    try:
        subprocess.run(["meshfix", str(src)], check=True)
    except FileNotFoundError:
        print(
            "[ERR] 'meshfix' binary not found in PATH.\n"
            "      Build it from https://github.com/alecjacobson/meshfix and make sure it is on PATH."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[ERR] meshfix failed on {src.name}: {exc}")
        return False

    # Convert OFF → STL if meshfix succeeded.
    if fixed_off.exists():
        try:
            import trimesh

            m = trimesh.load_mesh(fixed_off, force="mesh")
            m.export(dst, file_type="stl")
            fixed_off.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[ERR] Conversion OFF→STL failed for {src.name}: {exc}")
            return False
    else:
        print(
            f"[WARN] Expected output {fixed_off.name} not found – skipping conversion."
        )
        return False

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply meshfix to the remaining non-watertight Galaxea meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="Directory containing Galaxea STL meshes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original STL files after creating *.bak backups.",
    )
    return parser.parse_args()


def main() -> int:  # noqa: D103 – CLI wrapper
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[2]
    mesh_dir = args.mesh_dir or (
        project_root / "mani_skill" / "assets" / "robots" / "a1_galaxea" / "meshes"
    )

    if not mesh_dir.exists():
        print(f"[ERR] Mesh directory not found: {mesh_dir}")
        return 1

    success = 0
    for name in TARGET_FILES:
        src = mesh_dir / name
        if not src.exists():
            print(f"[WARN] File missing: {src}")
            continue

        if args.overwrite:
            backup = src.with_suffix(src.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(src, backup)
            dst = src  # overwrite in-place after meshfix
        else:
            dst = src.with_name(src.stem + "_meshfix" + src.suffix)

        if _run_meshfix(src, dst):
            success += 1

    print(f"\nSuccessfully processed {success}/{len(TARGET_FILES)} files.")
    return 0 if success == len(TARGET_FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
