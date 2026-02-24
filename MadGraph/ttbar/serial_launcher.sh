N=$1
for i in $(seq 1 "$N"); do
    madgraph launch.dat
done