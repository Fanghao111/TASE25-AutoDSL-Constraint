#!/bin/bash
for inst in "instance ta72" "instance ta80"; do
    dir="outputs/FB-CPE_CAE_CSE-2/$inst"
    for f in "$dir"/*; do
        base=$(basename "$f")
        if [ "$base" != "orders.json" ]; then
            rm "$f"
        fi
    done
    echo "Cleaned $dir (kept orders.json)"
done
