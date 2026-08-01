#!/bin/bash
cd "$(dirname "$0")/.."
git pull
echo "Pulled latest at $(date)"
