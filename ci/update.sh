#!/bin/bash
set -ex

REL_DIR_NAME=$(dirname "$0")
SCRIPT_DIR=$(cd "$REL_DIR_NAME" && pwd -P)
cd "$SCRIPT_DIR"/..

REPO_DIR="$PWD/wpt-results"

if [[ ! -e $REPO_DIR ]]; then
    git clone --bare https://github.com/jgraham/gecko-results-cache.git $REPO_DIR
fi

cd $REPO_DIR
git config user.email "wpt-results@mozilla.bugs"
git config user.name "wpt-results-bot"
git fetch '+refs/runs/*:refs/runs/*'
cd ..

uv run wpt-store-results --git-repo wpt-results --log-level debug --backfill=new

cd $REPO_DIR
git status
git push https://x-access-token:${GITHUB_TOKEN}@github.com/jgraham/gecko-results-cache.git '+refs/runs/*:refs/runs/*'
