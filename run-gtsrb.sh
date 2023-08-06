#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

#  array=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#  for e in "${array[@]}"
#  for e in $(seq 0.1 0.1 1)
#  for e in $(seq 0.01 0.01 0.09)
#  for e in $(seq 1)
model=("gtsrb-30x2-normal" "gtsrb-30x2-pgd")
for m in "${model[@]}"
do
  python gtsrb_incorrect_preds.py --network="$m" --epsilon=0.005 --output_path=gtsrb_outputs/"$m"
done
