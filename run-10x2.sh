#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

#  array=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#  for e in "${array[@]}"
#  for e in $(seq 0.1 0.1 1)
#  for e in $(seq 0.01 0.01 0.09)
#  for e in $(seq 1)
model=("mnist-10x2-normal" "mnist-10x2-pgd")
epsilon=(0.05)
for i in $(seq 100 1 149)
do
  for m in "${model[@]}"
  do
    python mnist_abstracted.py --network="$m" --epsilon="0.05" --index="$i" --output_path="$m"
  done
done
