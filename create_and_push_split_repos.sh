#!/bin/bash

# Script to create and push split ROMPY repos to GitHub org 'rom-py'
# Usage: bash create_and_push_split_repos.sh

set -e

ORG="rom-py"
REPOS=("rompy-core" "rompy-swan" "rompy-schism" "rompy-notebooks")
# REPOS=("rompy-swan" "rompy-schism")
#REPOS=("rompy")
SPLIT_DIR="../split-repos"

for REPO in "${REPOS[@]}"; do
    REPO_PATH="$SPLIT_DIR/$REPO"
    if [ ! -d "$REPO_PATH" ]; then
        echo "[SKIP] Directory $REPO_PATH does not exist. Skipping $REPO."
        continue
    fi
    echo "[INFO] Processing $REPO..."
    cd "$REPO_PATH"

    # Delete exiting repo if it exists remotely using gh with confirmation
    if gh repo view "$ORG/$REPO" > /dev/null 2>&1; then
        echo "[DELETE] Deleting existing remote repo $ORG/$REPO..."
        gh repo delete "$ORG/$REPO" 
        echo "[CREATE] Creating repo $ORG/$REPO on GitHub..."
        gh repo create "$ORG/$REPO" --public --confirm --description "Split from ROMPY monorepo. See https://github.com/rom-py/rompy-legacy for history."
    else
        echo "[INFO] Repo $ORG/$REPO does not exist on GitHub. Proceeding to create."
        gh repo create "$ORG/$REPO" --public --confirm --description "Split from ROMPY monorepo. See https://github.com/rom-py/rompy-legacy for history."
    fi

    # # Create repo in org if it doesn't exist
    # if ! gh repo view "$ORG/$REPO" > /dev/null 2>&1; then
    #     echo "[CREATE] Creating repo $ORG/$REPO on GitHub..."
    #     gh repo create "$ORG/$REPO" --public --confirm --description "Split from ROMPY monorepo. See https://github.com/rom-py/rompy-legacy for history."
    # else
    #     echo "[INFO] Repo $ORG/$REPO already exists on GitHub."
    # fi


    # Add remote if not present
    if ! git remote | grep -q origin; then
        echo "[REMOTE] Adding remote origin..."
        git remote add origin "git@github.com:$ORG/$REPO.git"
    fi

    # Ensure local branch is named 'main'
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        if git show-ref --verify --quiet refs/heads/main; then
            echo "[BRANCH] Local 'main' branch exists. Deleting..."
            git branch -D main
        fi
        echo "[BRANCH] Renaming $CURRENT_BRANCH to 'main'..."
        git branch -m main
    fi

    # Push local 'main' to remote 'main' (force if needed)
    echo "[PUSH] Pushing local 'main' to remote 'main'..."
    git push -u origin main

    echo "[DONE] $REPO pushed to https://github.com/$ORG/$REPO"
    cd - > /dev/null
    echo "------------------------------------------------------"
done

echo "All split repos processed. Please check for errors above."
