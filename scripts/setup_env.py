#!/usr/bin/env python3
"""Create `.env` from `.env.example` if it does not yet exist."""
import shutil, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
env_example = ROOT / ".env.example"
env_target  = ROOT / ".env"

if not env_example.exists():
    sys.exit("ERROR: .env.example missing – did you forget to pull it?")

if env_target.exists():
    print(".env already exists; no action taken.")
else:
    shutil.copy(env_example, env_target)
    print("Created .env from template; fill in your API keys!")