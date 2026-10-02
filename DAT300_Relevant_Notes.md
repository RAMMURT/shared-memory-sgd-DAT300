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

## Local experiment viewer

Run from the `shared-memory-sgd-DAT300` repository root with Python 3.10 or later:

```bash
python3 -m venv .venv-viewer
source .venv-viewer/bin/activate
python -m pip install -r requirements-viewer.txt
python experiment_viewer.py
```

Open one downloaded JSON file using **Open JSON** or **Ctrl+O**. You can also
pass a file on the command line:

```bash
python experiment_viewer.py ../experiments/2026-09-28_16-40-12_results_async.json
```

- **Overview:** accuracy milestones, recorded duration, acceptance rate, recorded
  staleness statistics, all configuration settings, and a field inventory.
- **Graphs:** scroll through accuracy by epoch/time, both loss series, staleness
  bars, epoch durations, and any recorded
  scheduling, per-epoch staleness, or step-time measurements. Use each toolbar
  to pan/zoom, or the save button to export PNG, SVG, or PDF. Staleness controls
  switch the histogram between counts/percentages and linear/log scale.
- **Raw:** full formatted JSON, with Previous/Next search and wraparound.

**Export all graphs** saves into a subdirectory named after the input file,
along with `summary.json`. GUI exports use the selected staleness display
options. Individual exports preserve the current zoom; export-all generates
the full figures. Filenames identify the metric, and each figure includes
the input filename, dispatcher, worker count, dataset, axis labels, and units.

Headless export is also available (no Qt window/display required):

```bash
python experiment_viewer.py ../experiments/2026-09-28_16-40-12_results_async.json --export /tmp/experiment-plots --format svg
```

CLI export replaces matching output files. GUI export-all asks before replacing
an existing nonempty export directory's matching files.

### Interpretation

- Accuracy is stored as a fraction and displayed as percent. Completed epochs
  start at 1. A dashed line labels the **first** epoch attaining the best recorded
  accuracy. `end_accuracy` is retained in Raw/field inventory; the overview uses
  the last epoch measurement because example files contain the sentinel `-1`.
- `epoch_loss` is monitor loss at completed epochs (1-based). `alt_epoch_loss`
  is the worker mean in `step_end_ind / epoch_steps` buckets (0-based); the last
  bucket can be partial. These are different measurements, not padded or shifted
  to force equal lengths.
- `tau_dist[i]` is the count for integer τ = i. Statistics use these counts as
  weights, never expand them into samples. Quantiles use the smallest integer τ
  at which cumulative probability reaches the requested quantile. Zero interior
  bins remain visible; trailing zero bins are trimmed. No smoothing or automatic
  claim of bimodality is applied. Log axes cannot display zero-height bins.
- The worker logger omits staleness beyond the histogram range. Mean, quantiles,
  and maximum describe **recorded samples**, not necessarily all updates.
- Timestamps ending in `mstimes` are cumulative milliseconds and are displayed
  in seconds. Recorded duration is the last epoch timestamp, not process runtime.
  The first epoch duration includes startup. Time to best accuracy uses that
  epoch's timestamp, not the accuracy evaluator's completion time.
- Optional step-time tuples are `[start, end, worker_id, tau]`. Their raw clock
  ticks are assumed to be nanoseconds for the Linux experiment platform; the
  graph labels state this assumption. Original tuples remain available in Raw.
- Empty/null measurements produce unavailable summaries or missing curves rather
  than fabricated zeros. Time plots use aligned entries when array lengths differ,
  with a note in Overview. Unknown fields remain visible in Raw/field inventory.

The viewer needs dependencies only on the local machine and does not contact the
remote server or alter the input JSON files.

Run the statistics, data-alignment, and export regression checks with:

```bash
python -m unittest discover -s tests -v
```
