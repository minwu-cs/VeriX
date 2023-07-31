#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

SECONDS=0
OUT_DIR=outputs/time-simple-cnn.txt

echo "Start timing simple cnn explanation..." >> $OUT_DIR
for i in $(seq 0 1 9)
do
  echo "Generating explanation for index $i)" >> $OUT_DIR
  python mnist_abstracted.py --network="mnist-simple-cnn-normal" --epsilon="0.1" --index="$i" --timeout=180
  echo "Explanation for index $i generated!" >> $OUT_DIR
  sleep 1
  echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec" >> $OUT_DIR
done