# DAT300 Relevant Notes

## Adding and evaluating a new SGD algorithm

The current implementation is assembled from modular components in `main.cpp`. Before adding code, decide which part of SGD the algorithm changes:

- **Synchronization or step scheduling:** implement a new `Dispatcher`. Existing examples are `AsyncDispatcher` and `SemiSyncDispatcher`.
- **Worker-side gradient computation or application:** extend the worker classes or the update path in `Component/sgdthread.cpp`.
- **Optimizer/update rule:** extend `Optimizer` or `ModelInterface` rather than putting optimizer logic in a dispatcher.
- **Dynamic worker count:** implement a `ParaController`. Existing examples include the static, ternary-search, and window controllers.
- **Data selection or batching:** implement a `BatchController`.
- **Measurements and evaluation:** implement or extend a `Monitor`.

The base interfaces are declared in `include/minidnn/modular_components.hpp`. New implementation files should follow the existing layout:

```text
include/minidnn/Component/NewAlgorithm.hpp
Component/new_algorithm.cpp
```

Rough implementation checklist:

1. Identify the smallest appropriate interface, usually `Dispatcher` for a new synchronization algorithm.
2. Implement all virtual methods required by that interface. For a dispatcher these are `try_start_step`, `finish_step`, and `is_finished`.
3. Keep shared counters and model state thread-safe. Study the atomic and locking behavior of the existing dispatchers before modifying it.
4. Add the new `.cpp` file to the `mininn` target in `CMakeLists.txt`.
5. Add a name for the algorithm in the component-selection logic in `main.cpp` (for example, another accepted value for `-D`).
6. If configuration is needed, add a short CLI flag to the `getopt` option string and switch in `main.cpp`, give it a documented default, and pass it to the component constructor.
7. Add the algorithm name and its important parameters to the JSON `meta` object in `main.cpp`. This is essential for identifying and reproducing results later.
8. Rebuild with `bash compile.sh`. Use `bash compile-debug.sh` while diagnosing concurrency or correctness problems.
9. First run a small sanity experiment with few threads, epochs, and steps. Check termination, finite loss, result-file creation, and the expected accepted/rejected step counts.
10. Add a reproducible comparison script under `test_scripts/` or `experiments/scripts/` and compare against at least one existing baseline using identical dataset, seed, batch size, learning rate, epoch count, and steps per epoch.

Example registration shape in `main.cpp`:

```cpp
if (o_dispatcher == "async") {
    exec.set_dispatcher(std::make_shared<AsyncDispatcher>(exec));
} else if (o_dispatcher == "new_algorithm") {
    exec.set_dispatcher(std::make_shared<NewAlgorithmDispatcher>(exec, algorithm_parameter));
} else {
    throw std::runtime_error("Unrecognised dispatcher name (-D)");
}
```

Example experiment layout:

```text
experiments/
├── scripts/
│   └── new_algorithm_vs_async.sh
├── README.md
└── results/          # generated JSON; do not commit large result sets
```

For fair experiments, run multiple repetitions and record the source revision, random seed, machine/thread configuration, dataset, all CLI arguments, and wall-clock date alongside the output. Vary one experimental factor at a time where possible. Useful existing output fields include epoch loss and timing, accuracy, staleness distribution, parallelism history, and step acceptance rate.

The files in `old_framework/` are not compiled into the current executable. They can be consulted for earlier implementations such as Leashed SGD, but new work should normally use the modular interfaces in `Component/` and `include/minidnn/Component/`.
