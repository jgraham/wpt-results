#!/bin/bash
set -ex

REL_DIR_NAME=$(dirname "$0")
SCRIPT_DIR=$(cd "$REL_DIR_NAME" && pwd -P)
cd "$SCRIPT_DIR"/..

REPO_DIR="$PWD/wpt-results"
REMOTE_REPO="https://github.com/jgraham/gecko-results-cache.git"

if [[ ! -e $REPO_DIR ]]; then
    git clone --bare "$REMOTE_REPO" "$REPO_DIR"
fi

cd $REPO_DIR
git config user.email "wpt-results@mozilla.bugs"
git config user.name "wpt-results-bot"
git fetch "$REMOTE_REPO" '+refs/runs/*:refs/runs/*'
cd ..

uv run wpt-store-results --git-repo wpt-results --log-level debug

cd $REPO_DIR
git push https://x-access-token:${GITHUB_TOKEN}@github.com/jgraham/gecko-results-cache.git '+refs/runs/*:refs/runs/*'
