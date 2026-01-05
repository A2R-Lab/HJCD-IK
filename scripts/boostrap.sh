#!/usr/bin/env bash
set -euo pipefail

# Rewrite git@github.com:* and git@github.com/* to https://github.com/*
git config url."https://github.com/".insteadOf "git@github.com:"
git config url."https://github.com/".insteadOf "git@github.com/"

git submodule sync --recursive
git submodule update --init --recursive
