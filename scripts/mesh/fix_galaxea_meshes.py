"""
Copyright 2025 Zordi, Inc. All rights reserved.

Repair non-watertight A1 Galaxea robot meshes using trimesh.

This utility walks through the Galaxea mesh directory (STL files) and attempts
basic automatic fixes so SAPIEN can build convex collision shapes.

Usage (from project root):

    # Dry-run – only tells you what would be fixed
    python scripts/mesh/fix_galaxea_meshes.py --dry-run

    # Overwrite meshes in-place after creating *.bak backups
    python scripts/mesh/fix_galaxea_meshes.py --backup

    # Write repaired copies next to originals with suffix "_repaired"
    python scripts/mesh/fix_galaxea_meshes.py --suffix _repaired

The script relies only on trimesh and pathlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _repair_mesh(mesh_path: Path) -> tuple[bool, trimesh.Trimesh]:
    """Attempt to repair *mesh_path* and return (repaired?, mesh).

    The routine performs common clean-ups and hole-filling steps that are
    usually enough to make CAD exports watertight.
    """
    mesh = trimesh.load_mesh(mesh_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        print(f"[SKIP] {mesh_path.name:20s} | not a surface mesh")
        return False, mesh  # type: ignore[return-value]

    # Track watertightness before modifications for statistics
    was_watertight = mesh.is_watertight

    # Basic clean-up
    # Replace deprecated helpers with the preferred APIs
    try:
        dup_mask = mesh.unique_faces()
        mesh.update_faces(dup_mask)
    except Exception:
        # unique_faces may not be available for some exotic meshes
        pass

    try:
        nondeg_mask = mesh.nondegenerate_faces()
        mesh.update_faces(nondeg_mask)
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()

    # Ensure consistent orientation & normals, fix inverted components
    try:
        mesh.process(validate=True)
    except Exception as exc:
        # Some pathological meshes trigger broadcasting errors inside Trimesh.
        # Fallback to a less strict validate=False pass; if that also fails we
        # continue without process() so we at least attempt hole-filling.
        print(
            f"[WARN] {mesh_path.name:20s} | mesh.process(validate=True) failed: {exc}. "
            "Trying validate=False."
        )
        try:
            mesh.process(validate=False)
        except Exception as exc2:
            print(
                f"[WARN] {mesh_path.name:20s} | mesh.process(validate=False) also failed: {exc2}. Skipping process()."
            )

    # Fill any remaining holes
    trimesh.repair.fill_holes(mesh)

    # In some cases normals are still inconsistent – attempt final fix
    mesh.fix_normals()

    is_watertight = mesh.is_watertight
    status = "ok" if is_watertight else "not-watertight"
    change_note = "repaired" if (not was_watertight and is_watertight) else status
    print(f"[DONE] {mesh_path.name:20s} | {change_note}")
    return is_watertight, mesh  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Repair A1 Galaxea STL meshes so they are watertight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="Path to directory containing Galaxea STL meshes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and attempt repair but **do not** write any files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backups before overwriting original files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix for repaired files. Empty string overwrites originals.",
    )
    return parser.parse_args()


def main() -> int:  # noqa: D103 – CLI wrapper
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[2]
    default_mesh_dir = (
        project_root / "mani_skill" / "assets" / "robots" / "a1_galaxea" / "meshes"
    )
    mesh_dir = args.mesh_dir or default_mesh_dir

    if not mesh_dir.exists():
        print(f"[ERR] Mesh directory not found: {mesh_dir}")
        return 1

    stl_files = sorted(list(mesh_dir.rglob("*.stl")) + list(mesh_dir.rglob("*.STL")))
    if not stl_files:
        print(f"[ERR] No STL files found in {mesh_dir}")
        return 1

    print(f"Repairing {len(stl_files)} STL meshes in {mesh_dir}…\n")
    repaired_total = 0
    for stl_path in stl_files:
        repaired, repaired_mesh = _repair_mesh(stl_path)
        if not args.dry_run and repaired:
            # Backup original if requested and overwriting in-place
            if args.backup and args.suffix == "":
                backup_path = stl_path.with_suffix(stl_path.suffix + ".bak")
                if not backup_path.exists():
                    backup_path.write_bytes(stl_path.read_bytes())

            target_path: Path
            if args.suffix:
                target_path = stl_path.with_name(
                    stl_path.stem + args.suffix + stl_path.suffix
                )
            else:
                target_path = stl_path

            repaired_mesh.export(target_path, file_type="stl")

        if repaired:
            repaired_total += 1

    print(f"\nRepaired {repaired_total}/{len(stl_files)} meshes successfully.")
    return 0 if repaired_total else 1


if __name__ == "__main__":
    sys.exit(main())
