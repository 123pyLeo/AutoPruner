"""Create a fresh, pinned OFT checkout and apply the recorded integration changes."""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

RESOURCES = "autoprune.integrations.openvla_oft"


def integration_manifest():
    return json.loads(files(RESOURCES).joinpath("manifest.json").read_text())


def verify_checkout(checkout):
    checkout = Path(checkout).resolve()
    manifest = integration_manifest()
    p = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if p.returncode or p.stdout.strip() != manifest["base_commit"]:
        raise RuntimeError("OFT checkout does not have the pinned upstream commit")
    invalid = []
    for entry in manifest["files"]:
        p = checkout / entry["path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry["release_sha256"]:
            invalid.append(entry["path"])
    if invalid:
        raise RuntimeError("OFT integration files differ from the reviewed overlay: " + ", ".join(invalid[:8]))
    return {"base_commit":manifest["base_commit"], "checked_files":len(manifest["files"])}


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new checkout directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = integration_manifest()
    patch = files(RESOURCES).joinpath("overlay.patch").read_bytes()
    if hashlib.sha256(patch).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Integration patch checksum mismatch")
    with tempfile.TemporaryDirectory(prefix=".autoprune-oft-", dir=destination.parent) as td:
        checkout = Path(td) / "checkout"
        checkout.mkdir()
        commands = [
            ["git", "init", "-q", str(checkout)],
            ["git", "-C", str(checkout), "fetch", "--depth=1", manifest["repository"], manifest["base_commit"]],
            ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
        ]
        for command in commands:
            subprocess.run(command, check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", "--check", "-"], input=patch, check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", "-"], input=patch, check=True)
        verify_checkout(checkout)
        shutil.move(str(checkout), str(destination))
    return verify_checkout(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_checkout(args.destination) if args.verify_only else prepare(args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
