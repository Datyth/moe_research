#!/usr/bin/env python3
"""Download Synapse/BTCV RawData via the Synapse API.

Reads SYNAPSE_TOKEN from a .env file at the project root (gitignored) and
syncs syn3193805 (Multi-Atlas Labeling Beyond the Cranial Vault) into
dataset/raw/synapse/. Requires an account with access already granted on
https://www.synapse.org/Synapse:syn3193805 (registration + Data Use
Agreement — done manually, not by this script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ENTITY = "syn3193805"
DEFAULT_DOWNLOAD_ROOT = PROJECT_ROOT / "dataset" / "raw" / "synapse"


def load_env_token(env_path: Path, *, key: str = "SYNAPSE_TOKEN") -> str:
    if not env_path.is_file():
        raise FileNotFoundError(f".env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        found_key, _, value = stripped.partition("=")
        if found_key.strip() == key:
            token = value.strip().strip('"').strip("'")
            if not token:
                raise ValueError(f"{key} is present in {env_path} but empty.")
            return token
    raise KeyError(f"{key} not found in {env_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    return parser.parse_args()


def main() -> None:
    import synapseclient
    import synapseutils

    args = parse_args()
    token = load_env_token(args.env_path)
    download_root = args.download_root.expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    print(f"Logged in to Synapse. Syncing {args.entity} -> {download_root}")

    synapseutils.syncFromSynapse(syn, args.entity, path=str(download_root))
    print("Download complete.")


if __name__ == "__main__":
    main()
