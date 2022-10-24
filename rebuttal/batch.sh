epsilon=(0.01)
model=("gtsrb-10x2")
traverse=('heuristic' 'random')
for t in "${traverse[@]}"
do
  for m in "${model[@]}"
  do
    for e in "${epsilon[@]}"
    do
      for i in $(seq 3 1 3)
      do
#        echo "$m", "$e", "$i", "$t"
        python gtsrb.py --network="$m" --epsilon="$e" --index="$i" --traverse="$t"
      done
    done
  done
done
