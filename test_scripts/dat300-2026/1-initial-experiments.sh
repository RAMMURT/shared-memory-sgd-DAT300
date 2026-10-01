./cmake-build/mininn -A CIFAR10 -D fully_sync -n 16 -e 5 -b 16 -l 0.005
./cmake-build/mininn -A CIFAR10 -D semisync -n 16 -e 5 -b 16 -l 0.005
./cmake-build/mininn -A CIFAR10 -D async -n 16 -e 5 -b 16 -l 0.005
