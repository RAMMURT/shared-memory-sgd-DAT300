<!-- readme based on template: https://github.com/othneildrew/Best-README-Template -->

<!-- PROJECT LOGO -->
<p align="center">
  <h1 align="center">shared-memory-sgd</h1>

  <p align="center">
    C++ framework for implementing shared-memory parallel SGD for Deep Neural Network training
    <br />
    <br />
    <a href="https://github.com/dcs-chalmers/shared-memory-sgd/issues">Report Bug</a>
    ·
    <a href="https://github.com/dcs-chalmers/shared-memory-sgd/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#input">Input</a></li>
        <li><a href="#output">Ouput</a></li>
        <li><a href="#examples">Examples</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Framework for implementing parallel shared-memory Artificial Neural Network (ANN) training in C++ with SGD, supporting various synchronization mechanisms degrees of consistency. The code builds upon the <a href="https://github.com/yixuan/MiniDNN">MiniDNN</a> implementation, and relies on Eigen and OpenMP. In particular, the project includes the implementation of **LEASHED** which guarantees consistency and lock-freedom.

For technical details of **LEASHED-SGD** and **ASAP.SGD** please see the original papers:

> Bäckström, K., Walulya, I., Papatriantafilou, M., & Tsigas, P. (2021, February). *Consistent Lock-free Parallel Stochastic Gradient Descent for Fast and Stable Convergence*. In Proceedings of the 35th IEEE International Parallel & Distributed Processing Symposium. <a href="https://arxiv.org/abs/2102.09032">Full version</a>.

> Bäckström, K., Papatriantafilou, M., & Tsigas, P. (2022, July). *ASAP-SGD: Instance-based Adaptiveness to Staleness in Asynchronous SGD*. In Proceedings of the 39th International Conference on Machine Learning *(to appear)*.

The following shared-memory parallel SGD algorithms are implemented:
* Lock-based consistent asynchronous SGD
* LEASHED - Lock-free implementation of consistent asynchronous SGD
* Hogwild! - Lock-free asynchronous SGD without consistency
* Synchronous parallel SGD

The following asynchrony-aware step size options are implemented:
* The TAIL-TAU Staleness-adaptive step size
* The FLeet staleness-adaptive step size <a href="https://dl.acm.org/doi/10.1145/3423211.3425685">[Damaskinos, G, et al. Middleware '20]</a>.
* Standard 1/staleness inverse step size scaling/dampening



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Dependencies

Eigen3 is required. It's expected to have its headers at /usr/include/eigen3 and /usr/include/eigen3/unsupported

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/dcs-chalmers/shared-memory-sgd.git
   ```
2. Build project
   ```sh
   bash build.sh
   ```
3. Compile
   ```sh
   bash compile.sh
   ```



<!-- USAGE EXAMPLES -->
## Usage

### Input

The executable accepts short options only. Every option requires a value.

Flag | Meaning | Values (default)
--- | --- | ---
`-A` | Dataset | `CIFAR10`, `CIFAR100`, `MNIST`, `FASHION-MNIST` (`CIFAR10`)
`-n` | Maximum worker threads | Integer (`10`)
`-l` | Learning rate | Float (`0.005`)
`-u` | Momentum | Float (`0`)
`-b` | Mini-batch size | Integer (`16`)
`-e` | Number of epochs | Integer (`500`)
`-s` | Steps per epoch | Integer (`3125`)
`-P` | Parallelism controller | `static`, `ternary`, `window`, `pattern`, `model` (`static`)
`-M` | Monitor | `window`, `ema`, `eval` (`window`)
`-D` | Dispatcher | `async`, `semisync`, `fully_sync` (`async`)
`-F` | Results directory | Path (`./experiments`)
`-p` | Probe steps for the `ternary` or `window` controller | Integer (`128`)
`-x` | Execution steps for the `ternary` or `window` controller | Integer (`512`)
`-d` | Search degree for the `ternary` controller | Integer (`2`)
`-w` | Search-window size for the `ternary` or `window` controller | Integer (`8`)
`-0` | Initial parallelism for the `window` controller | Integer (half of `-n`)
`-c` | Semi-sync period update strategy | `decay`, `probe`, `follow_m` (`decay`)
`-y` | Initial semi-sync period | Integer (`8000`)
`-q` | Semi-sync period reduction interval | Integer (`4096`)
`-z` | Semi-sync period reduction step | Integer (`0`)
`-m` | Minimum semi-sync period | Integer (`4`)
`-W` | Semi-sync probe-window offset | Integer (`16`)
`-S` | Semi-sync probe-window step | Integer (`4`)
`-L` | Semi-sync probe loss scalar | Float (`0.9`)

The network architecture is selected automatically from `-A`; there is no separate architecture flag. The current executable does not implement a `--help` option.

### Output

Output is a JSON object containing the following data:

Field | Meaning
--- | ---
`epoch_loss` | *list of loss values corresponding to each epoch*
`epoch_time` | *wall-clock time measure upon completing corresponding epoch*
`staleness_dist` | *distribution of staleness*
`numtriesdist` | *distribution of n.o. CAS attempts (applies to LSH only)*

### Examples

Asynchronous CIFAR-10 training for 5 epochs with 8 threads:
 ```sh
 ./cmake-build/mininn -A CIFAR10 -n 8 -e 5 -s 3125 -b 16 -l 0.005 -u 0.5 -P static -M window -D async
 ```

Fully synchronous Fashion-MNIST training for 10 epochs with 16 threads:
 ```sh
 ./cmake-build/mininn -A FASHION-MNIST -n 16 -e 10 -s 3750 -b 16 -l 0.002 -u 0.5 -P static -M window -D fully_sync
 ```

Semi-synchronous CIFAR-100 training using the decaying-period strategy:
 ```sh
 ./cmake-build/mininn -A CIFAR100 -n 32 -e 10 -s 3125 -b 16 -l 0.005 -u 0.5 -P static -M window -D semisync -c decay -y 256 -q 4096 -z 2 -m 4
 ```

Asynchronous CIFAR-10 training with the adaptive window parallelism controller, writing results to a custom directory:
 ```sh
 ./cmake-build/mininn -A CIFAR10 -n 64 -e 10 -s 3125 -b 16 -l 0.005 -u 0.3 -P window -0 32 -w 12 -p 1024 -x 8192 -M eval -D async -F ./experiments/window
 ```



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Reference the repository and the papers

 ```
@misc{backstrom2021framework,
  author = {Bäckström, Karl},
  title = {shared-memory-sgd},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dcs-chalmers/shared-memory-sgd}},
  commit = {XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX}
}
@inproceedings{backstrom2021consistent,
  title={Consistent lock-free parallel stochastic gradient descent for fast and stable convergence},
  author={B{\"a}ckstr{\"o}m, Karl and Walulya, Ivan and Papatriantafilou, Marina and Tsigas, Philippas},
  booktitle={2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={423--432},
  year={2021},
  organization={IEEE}
}
 ```



<!-- LICENSE -->
## License

Distributed under the AGPL-3.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Karl Bäckström - bakarl@chalmers.se

Project Link: [https://github.com/dcs-chalmers/shared-memory-sgd](https://github.com/dcs-chalmers/shared-memory-sgd)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

A big thanks to the Wallenberg AI, Autonomous Systems and Software Program (WASP) for funding this work.
