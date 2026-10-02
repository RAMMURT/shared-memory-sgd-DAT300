#!/usr/bin/env python3
"""Local, single-run experiment viewer; also supports headless graph export."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


SERIES = (
    "epoch_accuracy", "epoch_loss", "alt_epoch_loss", "epoch_mstimes",
    "async_period_values", "async_period_mstimes", "para_values",
    "para_mstimes", "para_stepinds",
)


class Experiment:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.data, dict) or not isinstance(self.data.get("meta"), dict):
            raise ValueError("Expected an experiment JSON object with a 'meta' object.")
        self.meta = self.data["meta"]
        self.notes = []
        for key in SERIES:
            self.array(key)
        self.tau = self.histogram(self.data.get("tau_dist", []), "tau_dist")
        self.epoch_tau = self.data.get("epoch_tau_dist", [])
        if not isinstance(self.epoch_tau, list):
            raise ValueError("epoch_tau_dist must be an array.")
        for i, row in enumerate(self.epoch_tau):
            self.histogram(row, f"epoch_tau_dist[{i}]")
        self.steps = self.data.get("steptimes", [])
        if not isinstance(self.steps, list):
            raise ValueError("steptimes must be an array.")
        for row in self.steps:
            if (not isinstance(row, list) or len(row) != 4
                    or any(not self.finite(x) for x in row) or row[1] < row[0]):
                raise ValueError("steptimes rows must contain [start, end, worker, tau], with end >= start.")
        self.check_pair("epoch_mstimes", "epoch_accuracy")
        self.check_pair("async_period_mstimes", "async_period_values")
        self.check_pair("para_mstimes", "para_values")
        accuracy = self.array("epoch_accuracy")
        if np.any(np.isfinite(accuracy) & ((accuracy < 0) | (accuracy > 1))):
            self.notes.append("Accuracy contains values outside [0, 1]; those values are excluded.")
        for key in SERIES:
            if np.any(~np.isfinite(self.array(key))):
                self.notes.append(f"{key}: missing/non-finite measurements appear as gaps.")
        times = self.array("epoch_mstimes")
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            self.notes.append("Epoch timestamps are negative or decrease; duration summaries are unavailable.")
        self.notes.append("τ statistics cover recorded samples only. The current logger omits τ outside the histogram range.")
        self.notes.append("Recorded duration ends at the last epoch timestamp; it is not a measured process completion time.")

    @staticmethod
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def array(self, key):
        values = self.data.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} must be an array.")
        if any(v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool)) for v in values):
            raise ValueError(f"{key} must contain numbers or null.")
        return np.array([float(v) if v is not None else np.nan for v in values])

    @staticmethod
    def histogram(values, name):
        if not isinstance(values, list) or any(
            not Experiment.finite(v) or v < 0 or int(v) != v for v in values
        ):
            raise ValueError(f"{name} must contain nonnegative integer counts.")
        return np.array(values, dtype=float)

    def check_pair(self, x, y):
        nx, ny = len(self.array(x)), len(self.array(y))
        if nx != ny and (nx or ny):
            self.notes.append(f"{x}/{y}: lengths differ ({nx}/{ny}); time plots use aligned entries only.")

    @property
    def accuracy(self):
        a = self.array("epoch_accuracy").copy()
        a[(a < 0) | (a > 1)] = np.nan
        return a

    @property
    def best_index(self):
        a = self.accuracy
        return int(np.nanargmax(a)) if np.any(np.isfinite(a)) else None

    @property
    def title(self):
        return f"{self.meta.get('dispatcher', 'Unknown method')} · {self.meta.get('num_threads', '?')} workers · {self.meta.get('dataset', 'Unknown dataset')}"

    def summary(self):
        rows = [("File", self.path.name), ("Run", self.title)]
        a, best = self.accuracy, self.best_index
        times = self.array("epoch_mstimes") / 1000
        valid_times = bool(len(times) and np.all(np.isfinite(times))
                           and np.all(times >= 0) and np.all(np.diff(times) >= 0))
        rows.append(("Recorded epochs", str(len(a))))
        rows.append(("Best accuracy", f"{a[best] * 100:.3f}% at epoch {best + 1}" if best is not None else "No recorded data"))
        rows.append(("Last recorded accuracy", f"{a[-1] * 100:.3f}%" if len(a) and np.isfinite(a[-1]) else "No recorded data"))
        rows.append(("Time to best accuracy", f"{times[best]:.3f} s" if valid_times and best is not None and best < len(times) else "Unavailable"))
        rows.append(("Recorded duration", f"{times[-1]:.3f} s" if valid_times else "Unavailable"))
        rate = self.data.get("step_acceptance_rate")
        rows.append(("Step acceptance rate", f"{rate * 100:.3f}%" if self.finite(rate) and 0 <= rate <= 1 else "Unavailable"))
        counts = self.tau
        total = counts.sum()
        rows.append(("Recorded τ samples", f"{int(total):,}"))
        if total:
            tau = np.arange(len(counts))
            mean = np.dot(tau, counts) / total
            std = np.sqrt(np.dot((tau - mean) ** 2, counts) / total)
            cdf = np.cumsum(counts) / total
            for name, value in (("Mean τ", f"{mean:.3f}"), ("Standard deviation τ", f"{std:.3f}"),
                                ("Median τ", str(np.searchsorted(cdf, .5))),
                                ("p95 τ", str(np.searchsorted(cdf, .95))),
                                ("p99 τ", str(np.searchsorted(cdf, .99))),
                                ("Modal τ (first if tied)", str(np.argmax(counts))),
                                ("Largest recorded τ", str(np.flatnonzero(counts)[-1])),
                                ("τ = 0", f"{counts[0] / total * 100:.3f}%")):
                rows.append((name, value))
        else:
            rows.append(("τ statistics", "No recorded data"))
        return rows


def figures(run, tau_percent=False, tau_log=False):
    """Return named figures using the same rendering for GUI and headless export."""
    output = []

    def chart(slug, title, xlabel, ylabel):
        fig = Figure(figsize=(10, 4.8), layout="constrained")
        ax = fig.subplots()
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.suptitle(f"{run.title}\n{run.path.name}", fontsize=10)
        ax.grid(True, alpha=.22)
        output.append((slug, fig))
        return ax

    def empty(ax):
        ax.text(.5, .5, "No recorded data", ha="center", va="center", transform=ax.transAxes)

    a = run.accuracy * 100
    ax = chart("accuracy_epoch", "Accuracy by epoch", "Epoch (1-based)", "Accuracy (%)")
    if np.any(np.isfinite(a)):
        ax.plot(np.arange(1, len(a) + 1), a, label="Epoch accuracy")
        i = run.best_index
        ax.axvline(i + 1, color="tab:red", linestyle="--", label=f"Best: epoch {i + 1}, {a[i]:.3f}%")
        ax.scatter([i + 1], [a[i]], color="tab:red", zorder=3)
        ax.legend()
    else:
        empty(ax)

    t = run.array("epoch_mstimes") / 1000
    if len(t) and len(a):
        ax = chart("accuracy_time", "Accuracy by elapsed time", "Elapsed time (s)", "Accuracy (%)")
        n = min(len(t), len(a))
        mask = np.isfinite(t[:n]) & (t[:n] >= 0) & np.isfinite(a[:n])
        ax.plot(np.where(mask, t[:n], np.nan), np.where(mask, a[:n], np.nan))
        i = run.best_index
        if i is not None and i < n and mask[i]:
            ax.axvline(t[i], color="tab:red", linestyle="--", label=f"Best: epoch {i + 1}, {a[i]:.3f}% at {t[i]:.2f} s")
            ax.legend()
        if not mask.any():
            empty(ax)

    ax = chart("loss_epoch", "Loss by epoch", "Epoch / epoch bucket", "Loss")
    has_loss = False
    for key, start, label in (("epoch_loss", 1, "Monitor loss (completed epochs, 1-based)"),
                              ("alt_epoch_loss", 0, "Worker mean loss (step-index buckets, 0-based; last may be partial)")):
        y = run.array(key)
        if np.any(np.isfinite(y)):
            ax.plot(np.arange(start, len(y) + start), y, label=label)
            has_loss = True
    if has_loss:
        ax.legend(fontsize=8)
    else:
        empty(ax)

    counts = run.tau
    total = counts.sum()
    # Retain every interior bin, including zeros between possible peaks.
    last = int(np.flatnonzero(counts)[-1]) + 1 if total else len(counts)
    x = np.arange(last)
    ax = chart("tau_histogram", "Recorded staleness distribution", "τ (parameter updates)", "Recorded samples (%)" if tau_percent else "Recorded sample count")
    if total:
        y = counts[:last] / total * 100 if tau_percent else counts[:last]
        ax.bar(x, y, width=.9)
        if tau_log:
            ax.set_yscale("log")
        ax.set_xlim(-.7, max(last - .3, 1))
    else:
        empty(ax)
    ax = chart("epoch_duration", "Time between epoch measurements", "Epoch (1-based)", "Duration (s)")
    if len(t):
        durations = np.diff(np.r_[0., t])
        durations[durations < 0] = np.nan
        ax.plot(np.arange(1, len(t) + 1), durations)
        ax.text(.01, .98, "First interval starts at run time zero and includes startup.", transform=ax.transAxes, va="top", fontsize=8)
    else:
        empty(ax)

    for prefix, title, ylabel in (("async_period", "Semisynchronous period schedule", "Period (steps)"),
                                  ("para", "Recorded parallelism schedule", "Workers")):
        y = run.array(prefix + "_values")
        x = run.array(prefix + "_mstimes") / 1000
        if len(y):
            n = min(len(x), len(y))
            ax = chart(prefix + "_schedule", title, "Elapsed time (s)" if n else "Record index (0-based)", ylabel)
            ax.step(x[:n] if n else np.arange(len(y)), y[:n] if n else y, where="post")
            if prefix == "para":
                probing = run.data.get("para_isprobing", [])
                if isinstance(probing, list) and n:
                    inds = [i for i in range(min(n, len(probing))) if probing[i] is True]
                    if inds:
                        ax.scatter(x[inds], y[inds], color="tab:orange", label="Probing")
                        ax.legend()

    if run.epoch_tau:
        width = max(map(len, run.epoch_tau), default=0)
        if width and any(sum(row) for row in run.epoch_tau):
            matrix = np.zeros((len(run.epoch_tau), width))
            for i, row in enumerate(run.epoch_tau):
                matrix[i, :len(row)] = row
            sums = matrix.sum(axis=1, keepdims=True)
            probabilities = np.divide(matrix * 100, sums, out=np.zeros_like(matrix), where=sums > 0)
            probabilities = np.ma.masked_where(np.broadcast_to(sums == 0, matrix.shape), probabilities)
            ax = chart("tau_by_epoch", "Recorded staleness by step-index epoch bucket", "τ (parameter updates)", "Epoch bucket (0-based)")
            mesh = ax.imshow(probabilities, aspect="auto", origin="lower", interpolation="nearest")
            ax.figure.colorbar(mesh, ax=ax, label="Recorded samples within bucket (%)")
    if run.steps:
        samples = np.asarray(run.steps, dtype=float)
        duration = (samples[:, 1] - samples[:, 0]) / 1e6
        ax = chart("step_duration", "Sampled worker step durations", "Duration (ms; logger clock ticks assumed ns)", "Sample count")
        ax.hist(duration, bins="auto")
        ax = chart("step_duration_tau", "Sampled step duration versus staleness", "τ (parameter updates)", "Duration (ms; logger clock ticks assumed ns)")
        ax.scatter(samples[:, 3], duration, s=8, alpha=.4)
    return output


def export_all(run, output_dir, fmt="png", **options):
    target = Path(output_dir) / run.path.stem
    target.mkdir(parents=True, exist_ok=True)
    for name, fig in figures(run, **options):
        fig.savefig(target / f"{name}.{fmt}", dpi=180)
        fig.clear()
    (target / "summary.json").write_text(json.dumps({"statistics": dict(run.summary()), "notes": run.notes, "meta": run.meta}, indent=2), encoding="utf-8")
    return target


def launch(path=None):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QKeySequence, QTextDocument
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QScrollArea, QLabel, QPushButton, QFileDialog,
        QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
        QPlainTextEdit, QLineEdit, QCheckBox, QComboBox, QLayout,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

    class Viewer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.run = None
            self.plot_widgets = []
            self.setWindowTitle("DAT300 Experiment Viewer")
            self.resize(1150, 850)
            center = QWidget()
            self.setCentralWidget(center)
            layout = QVBoxLayout(center)
            bar = QHBoxLayout()
            open_button = QPushButton("Open JSON…")
            open_button.clicked.connect(self.choose_file)
            bar.addWidget(open_button)
            self.filename = QLabel("Open a downloaded experiment JSON to begin.")
            self.filename.setTextFormat(Qt.TextFormat.PlainText)
            self.filename.setWordWrap(True)
            bar.addWidget(self.filename, 1)
            self.export_format = QComboBox()
            self.export_format.addItems(["png", "svg", "pdf"])
            bar.addWidget(self.export_format)
            self.export_button = QPushButton("Export all graphs…")
            self.export_button.setEnabled(False)
            self.export_button.clicked.connect(self.export)
            bar.addWidget(self.export_button)
            layout.addLayout(bar)
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)
            self.overview_layout = self.scroll_tab("Overview")
            self.overview_layout.addWidget(QLabel("No experiment loaded."))
            graphs = QWidget()
            graphs_layout = QVBoxLayout(graphs)
            controls = QHBoxLayout()
            self.percent = QCheckBox("τ histogram as percentage")
            self.log = QCheckBox("Logarithmic τ y-axis")
            for control in (self.percent, self.log):
                controls.addWidget(control)
                control.toggled.connect(self.render_graphs)
            controls.addStretch()
            graphs_layout.addLayout(controls)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            body = QWidget()
            self.graph_layout = QVBoxLayout(body)
            self.graph_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
            scroll.setWidget(body)
            graphs_layout.addWidget(scroll)
            self.tabs.addTab(graphs, "Graphs")
            raw = QWidget()
            raw_layout = QVBoxLayout(raw)
            search_bar = QHBoxLayout()
            self.search = QLineEdit()
            self.search.setPlaceholderText("Find in JSON…")
            self.search.returnPressed.connect(self.find_next)
            search_bar.addWidget(self.search)
            for label, backward in (("Previous", True), ("Next", False)):
                button = QPushButton(label)
                button.clicked.connect(lambda checked=False, back=backward: self.find_next(back))
                search_bar.addWidget(button)
            raw_layout.addLayout(search_bar)
            self.raw = QPlainTextEdit()
            self.raw.setReadOnly(True)
            self.raw.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.raw.setStyleSheet("font-family: monospace;")
            raw_layout.addWidget(self.raw)
            self.tabs.addTab(raw, "Raw")
            open_action = QAction("Open JSON", self)
            open_action.setShortcut(QKeySequence.StandardKey.Open)
            open_action.triggered.connect(self.choose_file)
            self.addAction(open_action)
            self.statusBar().showMessage("Ctrl+O to open a file. Each graph has zoom, pan, and save controls.")

        def scroll_tab(self, label):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            body = QWidget()
            layout = QVBoxLayout(body)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
            scroll.setWidget(body)
            self.tabs.addTab(scroll, label)
            return layout

        @staticmethod
        def clear(layout):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()

        def table(self, rows):
            table = QTableWidget(len(rows), 2)
            table.setHorizontalHeaderLabels(["Metric / setting", "Value"])
            table.verticalHeader().hide()
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            for i, (name, value) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(str(name)))
                table.setItem(i, 1, QTableWidgetItem(str(value)))
            table.resizeRowsToContents()
            table.setFixedHeight(sum(table.rowHeight(i) for i in range(len(rows))) + table.horizontalHeader().height() + 8)
            return table

        def choose_file(self):
            repo = Path(__file__).resolve().parent
            experiments = repo / "experiments"
            if not experiments.is_dir():
                experiments = repo.parent / "experiments"
            initial = str(self.run.path.parent if self.run else experiments)
            filename, _ = QFileDialog.getOpenFileName(self, "Open experiment", initial, "JSON files (*.json);;All files (*)")
            if filename:
                self.load(filename)

        def load(self, filename):
            try:
                candidate = Experiment(filename)
            except (OSError, ValueError, OverflowError) as error:
                QMessageBox.critical(self, "Cannot open experiment", str(error))
                return
            self.run = candidate
            self.filename.setText(str(candidate.path))
            self.setWindowTitle(f"{candidate.path.name} — DAT300 Experiment Viewer")
            self.raw.setPlainText(json.dumps(candidate.data, indent=2, ensure_ascii=False))
            self.clear(self.overview_layout)
            self.overview_layout.addWidget(QLabel("Run statistics"))
            self.overview_layout.addWidget(self.table(candidate.summary()))
            notes = QLabel("\n".join(candidate.notes))
            notes.setTextFormat(Qt.TextFormat.PlainText)
            notes.setWordWrap(True)
            self.overview_layout.addWidget(notes)
            self.overview_layout.addWidget(QLabel("All configuration settings"))
            self.overview_layout.addWidget(self.table(sorted(candidate.meta.items())))
            self.overview_layout.addWidget(QLabel("Recorded fields (full values in Raw)"))
            self.overview_layout.addWidget(self.table([(k, f"{len(v):,} entries" if isinstance(v, (list, dict)) else v) for k, v in candidate.data.items() if k != "meta"]))
            self.overview_layout.addStretch()
            self.render_graphs()
            self.export_button.setEnabled(True)
            self.statusBar().showMessage(f"Loaded {candidate.path.name}")

        def render_graphs(self):
            if self.run is None:
                return
            for _, canvas in self.plot_widgets:
                canvas.figure.clear()
            self.plot_widgets = []
            self.clear(self.graph_layout)
            for name, fig in figures(self.run, self.percent.isChecked(), self.log.isChecked()):
                card = QWidget()
                layout = QVBoxLayout(card)
                canvas = FigureCanvasQTAgg(fig)
                canvas.setMinimumHeight(460)
                toolbar = NavigationToolbar2QT(canvas, card)
                # Scroll the page with the wheel; use the toolbar for zooming.
                toolbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                save = QPushButton("Save graph as PNG / SVG / PDF…")
                save.clicked.connect(lambda checked=False, f=fig, slug=name: self.save_graph(f, slug))
                layout.addWidget(save)
                self.graph_layout.addWidget(card)
                self.plot_widgets.append((name, canvas))
            self.graph_layout.addStretch()

        def save_graph(self, fig, slug):
            filename, selected = QFileDialog.getSaveFileName(self, "Save graph", f"{self.run.path.stem}_{slug}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
            if filename:
                suffix = {"PNG (*.png)": ".png", "SVG (*.svg)": ".svg", "PDF (*.pdf)": ".pdf"}[selected]
                path = Path(filename)
                if not path.suffix:
                    path = path.with_suffix(suffix)
                    if path.exists() and QMessageBox.question(self, "Replace graph?", f"Replace {path.name}?") != QMessageBox.StandardButton.Yes:
                        return
                try:
                    fig.savefig(path, dpi=180)
                    self.statusBar().showMessage(f"Saved {path}")
                except (OSError, ValueError) as error:
                    QMessageBox.critical(self, "Export failed", str(error))

        def export(self):
            folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
            if folder:
                target = Path(folder) / self.run.path.stem
                if target.exists() and any(target.iterdir()):
                    if QMessageBox.question(self, "Replace exported graphs?", f"Existing matching exports in {target} will be replaced. Continue?") != QMessageBox.StandardButton.Yes:
                        return
                try:
                    target = export_all(self.run, folder, self.export_format.currentText(), tau_percent=self.percent.isChecked(), tau_log=self.log.isChecked())
                    self.statusBar().showMessage(f"Exported graphs and summary to {target}")
                except (OSError, ValueError) as error:
                    QMessageBox.critical(self, "Export failed", str(error))

        def find_next(self, backward=False):
            text = self.search.text()
            if not text:
                return
            flags = QTextDocument.FindFlag.FindBackward if backward else QTextDocument.FindFlag(0)
            if not self.raw.find(text, flags):
                cursor = self.raw.textCursor()
                cursor.movePosition(cursor.MoveOperation.End if backward else cursor.MoveOperation.Start)
                self.raw.setTextCursor(cursor)
                if not self.raw.find(text, flags):
                    self.statusBar().showMessage("Search text not found.")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("DAT300 Experiment Viewer")
    window = Viewer()
    if path:
        window.load(path)
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="One experiment JSON file")
    parser.add_argument("--export", type=Path, metavar="DIRECTORY", help="Export graphs and summary without opening a window")
    parser.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
    args = parser.parse_args()
    if args.export:
        if args.file is None:
            parser.error("--export requires a JSON file")
        try:
            print(export_all(Experiment(args.file), args.export, args.format))
        except (OSError, ValueError, OverflowError) as error:
            parser.exit(1, f"Cannot export experiment: {error}\n")
        return 0
    return launch(args.file)


if __name__ == "__main__":
    sys.exit(main())
