#!/usr/bin/env bash
# Push the Train-on-Boot Space to HF.
#
# Prereqs (do these once, in your terminal):
#   pip install --upgrade huggingface_hub
#   huggingface-cli login            # write-access token
#
#   In the HF UI, create the Space:
#     https://huggingface.co/new-space
#     - Owner: <your hf user>
#     - Name:  susint-colab
#     - SDK:   Docker
#   Then go Settings → Hardware → T4 small (or larger).
#   Settings → Sleep time → "Never" so training isn't interrupted.
#
# Run from repo root:
#   bash space-colab/deploy.sh Hollow-Abyss/susint-colab
set -euo pipefail

REPO="${1:?usage: bash space-colab/deploy.sh <hf-user>/<space-name>}"

STAGE="$(mktemp -d)"
echo "[deploy] staging in $STAGE"
cp -R space-colab/. "$STAGE/"

cd "$STAGE"
# The Space dir's deploy.sh shouldn't ship into the Space itself.
rm -f deploy.sh

git init -q
git remote add origin "https://huggingface.co/spaces/${REPO}"
git checkout -b main 2>/dev/null || git checkout main
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "deploy susint train-on-boot space"
echo "[deploy] pushing to https://huggingface.co/spaces/${REPO}"
git push -u origin main --force
echo "[deploy] done."
echo
echo "Next:"
echo "  1. Open the Space and watch the build (~10-20 min for torch/unsloth)."
echo "  2. After build, training auto-starts in a background thread."
echo "  3. Poll status: curl https://${REPO/\//-}.hf.space/training/status"
echo "  4. View plot when done: https://${REPO/\//-}.hf.space/training/plot.png"
