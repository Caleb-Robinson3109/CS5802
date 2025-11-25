#!/bin/bash

SOURCE="./../img2.bmp"
DEST_DIR="../img2"
COPIES=100

mkdir -p "$DEST_DIR"

FILENAME=$(basename -- "$SOURCE")
NAME="${FILENAME%.*}"
EXT="${FILENAME##*.}"

for((i=1; i<=COPIES; i++)); do
    cp "$SOURCE" "$DEST_DIR/${NAME}_$i.$EXT"
done

echo "made $COPIES copies of $SOURCE in $DEST_DIR/"