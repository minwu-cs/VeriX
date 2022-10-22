#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in $(seq 0 1 99)
do
  for e in "0.1"
  do
#    echo "$i", "$e"
    python mnist.py --index="$i" --epsilon="$e"
  done
done


#for i in $(seq 0 1 49)
#do
#  for e in "0.03"
#  do
#    echo "$i", "$e"
#    python gtsrb.py --index="$i" --epsilon="$e"
#  done
#done

echo "Done."
