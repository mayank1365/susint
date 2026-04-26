#!/usr/bin/env bash
# Deploys the Suspect X environment to a Hugging Face Space.
#
# One-time prerequisites (do these in your terminal):
#   pip install --upgrade huggingface_hub
#   huggingface-cli login            # paste a write-access token
#   # Then in the HF UI, create a new Space:
#   #   https://huggingface.co/new-space
#   #   - SDK: Docker
#   #   - Name: suspect-x-env (or whatever)
#
# Then run this script from the repo root:
#   bash space/deploy.sh <hf-username>/<space-name>
#
# Example:
#   bash space/deploy.sh hivexlabs/suspect-x-env
set -euo pipefail

REPO="${1:?usage: bash space/deploy.sh <hf-username>/<space-name>}"
SPACE_DIR="$(mktemp -d)"
echo "[deploy] staging in $SPACE_DIR"

# Stage just the files HF needs.
cp space/Dockerfile  "$SPACE_DIR/Dockerfile"
cp space/README.md   "$SPACE_DIR/README.md"
cp -R suspect_x_env/server "$SPACE_DIR/server"
cp -R descriptions          "$SPACE_DIR/descriptions"

cd "$SPACE_DIR"
git init -q
git lfs install >/dev/null 2>&1 || true
git remote add origin "https://huggingface.co/spaces/${REPO}"
git checkout -b main 2>/dev/null || git checkout main
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "deploy suspect_x_env"
echo "[deploy] pushing to https://huggingface.co/spaces/${REPO}"
git push -u origin main --force
echo "[deploy] done. Space build will start in HF UI within ~30s."
