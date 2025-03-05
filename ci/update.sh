#!/bin/bash
set -ex

REL_DIR_NAME=$(dirname "$0")
SCRIPT_DIR=$(cd "$REL_DIR_NAME" && pwd -P)
cd "$SCRIPT_DIR"/..

git config --global user.email "wpt-results@mozilla.bugs"
git config --global user.name "wpt-results-bot"

uv run wpt-store-results --git-repo wpt-results --log-level debug --backfill=new
cd wpt-results
git status
git log
git push https://x-access-token:${GITHUB_TOKEN}@github.com/jgraham/gecko-wpt-results.git HEAD:main
