OUT_DIR=experiments/dat300/experiment6/a-2.5
mkdir -p $OUT_DIR

./cmake-build/mininn -A CIFAR10 -D semisync -n 16 -u 0.5 -z 2 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D semisync -n 32 -u 0.5 -z 2 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D semisync -n 48 -u 0.5 -z 2 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D semisync -n 64 -u 0.5 -z 2 -e 5 -b 16 -l 0.005 -F $OUT_DIR
