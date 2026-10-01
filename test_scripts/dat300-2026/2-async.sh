OUT_DIR=experiments/dat300/experiment4
mkdir -p $OUT_DIR

./cmake-build/mininn -A CIFAR10 -D async -n 16 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D async -n 32 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D async -n 48 -e 5 -b 16 -l 0.005 -F $OUT_DIR
./cmake-build/mininn -A CIFAR10 -D async -n 64 -e 5 -b 16 -l 0.005 -F $OUT_DIR
