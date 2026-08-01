#!/bin/bash
# Run before committing checkpoints — warns on anything too big for plain git
echo "Checking checkpoint sizes..."
find checkpoints/ -type f -size +50M | while read -r f; do
    echo "WARNING: $f exceeds 50MB — do not commit to plain git, keep cluster-only"
done
echo "Done."
