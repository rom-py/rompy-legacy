#!/bin/bash

# Script to create and push split ROMPY repos to GitHub org 'rom-py'
# Usage: bash create_and_push_split_repos.sh

set -e

ORG="rom-py"
REPOS=("rompy-core" "rompy-swan" "rompy-schism" "rompy-notebooks")
SPLIT_DIR="/home/tdurrant/source/rompy/split-repos"

for REPO in "${REPOS[@]}"; do
    REPO_PATH="$SPLIT_DIR/$REPO"
    if [ ! -d "$REPO_PATH" ]; then
        echo "[SKIP] Directory $REPO_PATH does not exist. Skipping $REPO."
        continue
    fi
    echo "[INFO] Processing $REPO..."
    cd "$REPO_PATH"

    # Create repo in org if it doesn't exist
    if ! gh repo view "$ORG/$REPO" > /dev/null 2>&1; then
        echo "[CREATE] Creating repo $ORG/$REPO on GitHub..."
        gh repo create "$ORG/$REPO" --public --confirm --description "Split from ROMPY monorepo. See https://github.com/rom-py/rompy for history."
    else
        echo "[INFO] Repo $ORG/$REPO already exists on GitHub."
    fi

    # Init git if needed
    if [ ! -d ".git" ]; then
        echo "[INIT] Initializing git repo in $REPO_PATH..."
        git init
        git checkout -b main
        git add .
        git commit -m "Initial commit: split from ROMPY monorepo"
    fi

    # Add remote if not present
    if ! git remote | grep -q origin; then
        echo "[REMOTE] Adding remote origin..."
        git remote add origin "https://github.com/$ORG/$REPO.git"
    fi

    # Push to main (force if needed)
    echo "[PUSH] Pushing $REPO to GitHub..."
    git push -u origin main --force

    echo "[DONE] $REPO pushed to https://github.com/$ORG/$REPO"
    cd - > /dev/null
    echo "------------------------------------------------------"
done

echo "All split repos processed. Please check for errors above."
