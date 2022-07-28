#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

#for i in $(seq 17 1 19)
#do
##  array=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
##  for e in "${array[@]}"
#  for e in $(seq 0.1 0.1 1)
##  for e in $(seq 0.01 0.01 0.09)
##  for e in $(seq 1)
##  for e in "0.9"
#  do
##    echo "$i", "$e"
#    python explanation_softmax.py --index="$i" --epsilon="$e"
#  done
#done



#for i in $(seq 8 1 9)
#do
#  for random in $(seq 0 1 99)
#  do
#    python explanation_random.py --index="$i" --epsilon=0.1
#  done
#done

for i in $(seq 0 1 49)
do
  for e in "0.01"
  do
#    echo "$i", "$e"
    python gtsrb.py --index="$i" --epsilon="$e"
  done

done

echo "Done."
