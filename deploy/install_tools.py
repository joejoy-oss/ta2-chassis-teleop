#!/usr/bin/env python3
"""Copy the archived workbench code to B70; never start services or robot motion."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("/data/ta2"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    mappings = [
        (repo / "tools", args.target / "tools"),
        (repo / "home", args.target / "home"),
        (repo / "perception/head-depth", args.target / "head-depth"),
        (repo / "perception/head-detection", args.target / "head-detection"),
        (repo / "deploy/config/perception", args.target / "perception"),
        (repo / "deploy/config/grasp", args.target / "grasp"),
    ]
    copied = preserved = 0
    for source, destination in mappings:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if "__pycache__" in relative.parts or not path.is_file() or path.suffix == ".pyc":
                continue
            out = destination / relative
            # Site calibration and service configuration belong to the operator.
            if out.exists() and (path.suffix == ".json" or path.name.endswith(".env")):
                preserved += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied += 1
    print(f"Copied {copied} files to {args.target}; preserved {preserved} existing configurations.")
    print("No service units installed, services restarted, models loaded, or robot commands sent.")


if __name__ == "__main__":
    main()
