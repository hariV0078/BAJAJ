#!/bin/bash
set -o errexit

# Install Rust without modifying system paths
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
source "$CARGO_HOME/env"

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
