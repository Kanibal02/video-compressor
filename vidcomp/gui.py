"""PySide6 desktop app for the video compressor."""
from __future__ import annotations

import ctypes
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, QUrl, QMimeData, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QKeySequence, QPainter, QPalette,
                           QIcon, QPixmap, QPolygon)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter, QStyle,
                               QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from . import engine as E

APP_NAME = "Video Compressor"
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(APP_DIR, "settings.json")
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets").replace("\\", "/")

ACCENT = "#6d5cff"
QUALITY_COLORS = {"Excellent": "#4ade80", "Good": "#a3e635", "OK": "#facc15",
                  "Low": "#fb923c", "Very low": "#f87171", "CQ": "#93c5fd"}

CODEC_TIPS = {
    "h264": ("H.264 - plays everywhere: every browser, phone and Discord client.\n"
             "Needs the most bits for the same quality."),
    "hevc": ("H.265 - ~25% smaller than H.264 at the same quality.\n"
             "Plays on phones, Macs and PCs with a hardware HEVC decoder (most GPUs since ~2016).\n"
             "Some browsers / very old PCs can't play it."),
    "av1": ("AV1 - best quality per MB (~40% better than H.264).\n"
            "Discord desktop and Chrome decode it in software if the GPU can't, so it plays there;\n"
            "older phones (esp. older iPhones) and some browsers may not play it."),
}

COL_FILE, COL_SRC, COL_PLAN, COL_TRIM, COL_PROG, COL_RESULT = range(6)
HEADERS = ["File", "Source", "Output plan", "Trim", "Progress", "Result"]


def complexity_label(bpp: float | None) -> str:
    if bpp is None:
        return "unknown (not analysed)"
    if bpp < 0.012:
        return f"Low motion ({bpp:.3f})"
    if bpp < 0.03:
        return f"Medium ({bpp:.3f})"
    if bpp < 0.055:
        return f"High ({bpp:.3f})"
    return f"Very high ({bpp:.3f})"


# --------------------------------------------------------------------------- model

class Job:
    _ids = itertools.count(1)

    def __init__(self, path: str):
        self.id = next(Job._ids)
        self.path = path
        self.name = os.path.basename(path)
        self.info: E.MediaInfo | None = None
        self.complexity: float | None = None
        self.analysed = threading.Event()
        self.probing_complexity = False
        self.trim: tuple[float | None, float | None] = (None, None)
        self.status = "analysing"   # analysing | ready | queued | encoding | done | failed | cancelled | skipped
        self.progress = 0.0
        self.speed: float | None = None
        self.stage = ""
        self.plan: E.Plan | None = None
        self.result: E.EncodeResult | None = None
        self.error = ""
        self.output = ""
        self.token: E.CancelToken | None = None
        self.started = 0.0


class Bus(QObject):
    job_changed = Signal(int)
    job_progress = Signal(int)
    run_finished = Signal()
    encoders_ready = Signal(list)


# --------------------------------------------------------------------------- widgets

class ProgressDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index):
        data = index.data(Qt.UserRole) or (None, index.data() or "", "#3b3f4c")
        frac, text, color = data
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        painter.setRenderHint(QPainter.Antialiasing)
        r = option.rect.adjusted(6, 6, -6, -6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#262833"))
        painter.drawRoundedRect(r, 6, 6)
        if frac is not None and frac > 0:
            fr = r.adjusted(0, 0, -int(r.width() * (1 - min(1.0, frac))), 0)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(fr, 6, 6)
        painter.setPen(QColor("#f1f1f6"))
        painter.drawText(r, Qt.AlignCenter, text)
        painter.restore()


class LabeledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, left: str, right: str, lo=0, hi=100, fmt=None, parent=None):
        super().__init__(parent)
        self.fmt = fmt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.valueChanged.connect(self._changed)
        row = QHBoxLayout()
        l1, l2 = QLabel(left), QLabel(right)
        for l in (l1, l2):
            l.setObjectName("hint")
        self.mid = QLabel()
        self.mid.setAlignment(Qt.AlignCenter)
        self.mid.setObjectName("sliderValue")
        row.addWidget(l1)
        row.addStretch()
        row.addWidget(self.mid)
        row.addStretch()
        row.addWidget(l2)
        lay.addWidget(self.slider)
        lay.addLayout(row)

    def _changed(self, v):
        if self.fmt:
            self.mid.setText(self.fmt(v))
        self.valueChanged.emit(v)

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(int(v))
        if self.fmt:
            self.mid.setText(self.fmt(int(v)))


class DropTable(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(0, len(HEADERS))
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.empty = QLabel("Drop videos or folders here\n\nor use  + Add files", self.viewport())
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setObjectName("emptyHint")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.empty.resize(self.viewport().size())

    def update_empty(self):
        self.empty.setVisible(self.rowCount() == 0)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            e.acceptProposedAction()


class TrimDialog(QDialog):
    def __init__(self, parent, job: Job):
        super().__init__(parent)
        self.setWindowTitle(f"Trim - {job.name}")
        dur = job.info.duration if job.info else 0
        lay = QFormLayout(self)
        self.start = QLineEdit("" if job.trim[0] is None else E.fmt_dur(job.trim[0]) if job.trim[0] == int(job.trim[0]) else f"{job.trim[0]:.2f}")
        self.end = QLineEdit("" if job.trim[1] is None else E.fmt_dur(job.trim[1]) if job.trim[1] == int(job.trim[1]) else f"{job.trim[1]:.2f}")
        self.start.setPlaceholderText("0:00")
        self.end.setPlaceholderText(E.fmt_dur(dur))
        lay.addRow(QLabel(f"Duration: {E.fmt_dur(dur)}   (formats: 75, 1:15, 0:01:15.5)"))
        lay.addRow("Start", self.start)
        lay.addRow("End", self.end)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.dur = dur
        self.value = job.trim

    def _ok(self):
        try:
            s = E.parse_time(self.start.text())
            e = E.parse_time(self.end.text())
        except ValueError as ex:
            QMessageBox.warning(self, "Trim", str(ex))
            return
        if s is not None and s >= (e if e is not None else self.dur):
            QMessageBox.warning(self, "Trim", "Start must be before end.")
            return
        self.value = (s if s else None, e if e is not None and e < self.dur - 0.01 else None)
        self.accept()


# --------------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        self.resize(1400, 820)
        self.setAcceptDrops(True)

        self.settings = E.Settings()
        self.ui_state = {"play_sound": True, "open_folder_when_done": False, "splitter": None}
        self._load_config()

        self.bus = Bus()
        self.bus.job_changed.connect(self._refresh_job)
        self.bus.job_progress.connect(self._refresh_progress)
        self.bus.run_finished.connect(self._on_run_finished)
        self.bus.encoders_ready.connect(self._on_encoders_ready)

        self.jobs: dict[int, Job] = {}
        self.analysis_pool = ThreadPoolExecutor(2)
        self.running = False
        self.run_jobs: list[Job] = []
        self.run_started = 0.0
        self.run_settings: E.Settings | None = None
        self._reserved: set[str] = set()
        self._reserve_lock = threading.Lock()

        self._binding = False
        self._avail_names: set[str] | None = None
        self._replan_timer = QTimer(self, singleShot=True, interval=120, timeout=self._replan_all)
        self._save_timer = QTimer(self, singleShot=True, interval=800, timeout=self._save_config)
        self._overall_timer = QTimer(self, interval=500, timeout=self._update_overall)

        self._build_ui()
        self._load_widgets()
        self._update_visibility()

        threading.Thread(target=lambda: self.bus.encoders_ready.emit(E.available_encoders()),
                         daemon=True).start()
        if not E.ffmpeg_available():
            QTimer.singleShot(200, lambda: QMessageBox.critical(
                self, APP_NAME, "ffmpeg / ffprobe were not found on PATH.\n\n"
                "Install ffmpeg (e.g. `winget install Gyan.FFmpeg`) and restart."))

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter, 1)

        # ---- left: queue
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        bar = QHBoxLayout()
        title = QLabel("Queue")
        title.setObjectName("title")
        bar.addWidget(title)
        bar.addStretch()
        for text, slot in (("+ Add files", self._add_files_dialog), ("+ Add folder", self._add_folder_dialog),
                           ("Clear finished", self._clear_finished), ("Clear all", self._clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        ll.addLayout(bar)

        self.table = DropTable()
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.files_dropped.connect(self.add_paths)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setItemDelegateForColumn(COL_PROG, ProgressDelegate(self.table))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        for c, w in ((COL_SRC, 175), (COL_PLAN, 250), (COL_TRIM, 70), (COL_PROG, 180), (COL_RESULT, 120)):
            hh.setSectionResizeMode(c, QHeaderView.Interactive)
            self.table.setColumnWidth(c, w)
        hh.setHighlightSections(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.doubleClicked.connect(self._double_clicked)
        ll.addWidget(self.table, 1)
        self.table.update_empty()

        dele = QAction(self)
        dele.setShortcut(QKeySequence.Delete)
        dele.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        dele.triggered.connect(self._remove_selected)
        self.table.addAction(dele)

        self.splitter.addWidget(left)

        # ---- right: settings
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(400)
        self.tabs.addTab(self._scroll(self._tab_target()), "Target")
        self.tabs.addTab(self._scroll(self._tab_video()), "Res / FPS")
        self.tabs.addTab(self._scroll(self._tab_encoder()), "Encoder")
        self.tabs.addTab(self._scroll(self._tab_audio()), "Audio")
        self.tabs.addTab(self._scroll(self._tab_output()), "Output")
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        if self.ui_state.get("splitter"):
            self.splitter.setSizes(self.ui_state["splitter"])
        else:
            self.splitter.setSizes([960, 440])

        # ---- bottom bar
        bottom = QHBoxLayout()
        self.status = QLabel("Add some videos to get started.")
        self.status.setObjectName("hint")
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setTextVisible(False)
        self.overall.setFixedHeight(8)
        stat_col = QVBoxLayout()
        stat_col.setSpacing(6)
        stat_col.addWidget(self.status)
        stat_col.addWidget(self.overall)
        bottom.addLayout(stat_col, 1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.start_btn = QPushButton("Compress")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.start_btn.setMinimumWidth(150)
        self.turbo_btn = QPushButton("⚡ Turbo")
        self.turbo_btn.setObjectName("turbo")
        self.turbo_btn.setCheckable(True)
        self.turbo_btn.setToolTip(
            "Fastest encoder settings, slightly lower quality at the same file size.\n"
            "Measured on a 1440p60 clip (OBS running):\n"
            "  H.264: 2.5x → 4.4x realtime, a bit softer in fast motion\n"
            "  AV1:   2.3x → 4.1x realtime, practically no visible difference\n"
            "  HEVC:  → 4.1x realtime")
        self.turbo_btn.toggled.connect(self._turbo_toggled)
        # quick codec picker - mirrors the Encoder tab's dropdown
        self.codec_group = QButtonGroup(self)
        self.codec_group.setExclusive(True)
        self.codec_btns: dict[str, QPushButton] = {}
        codec_row = QHBoxLayout()
        codec_row.setSpacing(0)
        for i, (codec, text) in enumerate((("h264", "H.264"), ("hevc", "H.265"), ("av1", "AV1"))):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setObjectName("segL" if i == 0 else "segR" if i == 2 else "segM")
            b.setToolTip(CODEC_TIPS[codec])
            b.clicked.connect(lambda _=False, c=codec: self._pick_codec(c))
            self.codec_group.addButton(b)
            self.codec_btns[codec] = b
            codec_row.addWidget(b)
        bottom.addSpacing(12)
        bottom.addLayout(codec_row)
        bottom.addSpacing(8)
        bottom.addWidget(self.turbo_btn)
        bottom.addWidget(self.stop_btn)
        bottom.addWidget(self.start_btn)
        outer.addLayout(bottom)

    def _scroll(self, w: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        return sa

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)
        return w, lay

    def _group(self, title: str, lay: QVBoxLayout) -> QFormLayout:
        g = QGroupBox(title)
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        f.setHorizontalSpacing(14)
        f.setVerticalSpacing(8)
        lay.addWidget(g)
        return f

    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QComboBox:
        c = QComboBox()
        for label, data in items:
            c.addItem(label, data)
        return c

    @staticmethod
    def _hint(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("hint")
        l.setWordWrap(True)
        return l

    def _tab_target(self):
        w, lay = self._page()
        f = self._group("Target", lay)
        self.w_target_mode = self._combo([("File size", "size"), ("% of original size", "percent"),
                                          ("Bitrate (total)", "bitrate"), ("Constant quality (no size limit)", "quality")])
        f.addRow("Mode", self.w_target_mode)

        self.w_size = QDoubleSpinBox()
        self.w_size.setRange(0.5, 100000)
        self.w_size.setDecimals(1)
        self.w_size.setSingleStep(1)
        self.w_unit = self._combo([("MB", "MB"), ("MiB", "MiB")])
        self.w_unit.setToolTip("MB = 1,000,000 bytes (safer), MiB = 1,048,576 bytes")
        row = QHBoxLayout()
        row.addWidget(self.w_size, 1)
        row.addWidget(self.w_unit)
        self.row_size = QWidget()
        self.row_size.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("Size", self.row_size)
        presets = QHBoxLayout()
        presets.setContentsMargins(0, 0, 0, 0)
        presets.setSpacing(4)
        for mb in (8, 10, 25, 50, 100, 500):
            b = QPushButton(f"{mb}")
            b.setObjectName("chip")
            b.setToolTip({10: "Discord free", 50: "Discord Nitro Basic", 500: "Discord Nitro"}.get(mb, ""))
            b.clicked.connect(lambda _=False, v=mb: (self.w_size.setValue(v), self.w_target_mode.setCurrentIndex(0)))
            presets.addWidget(b)
        self.row_presets = QWidget()
        self.row_presets.setLayout(presets)
        f.addRow("Presets", self.row_presets)

        self.w_percent = QDoubleSpinBox()
        self.w_percent.setRange(1, 100)
        self.w_percent.setSuffix(" %")
        f.addRow("Percent", self.w_percent)
        self.w_bitrate = QSpinBox()
        self.w_bitrate.setRange(100, 200000)
        self.w_bitrate.setSingleStep(250)
        self.w_bitrate.setSuffix(" kbps")
        f.addRow("Bitrate", self.w_bitrate)
        self.w_cq = QSpinBox()
        self.w_cq.setRange(10, 51)
        self.w_cq.setToolTip("Lower = better quality & bigger file. ~23-28 is visually good.")
        f.addRow("Quality (CQ/CRF)", self.w_cq)

        g = self._group("How to hit the target", lay)
        self.w_priority = LabeledSlider("Keep smoothness (fps)", "Keep detail (resolution)",
                                        fmt=lambda v: "balanced" if 40 <= v <= 60 else f"{v}")
        g.addRow(self.w_priority)
        g.addRow(self._hint("When the budget is tight, which do you sacrifice first? "
                            "Left keeps 60 fps and lowers resolution, right keeps 1440p and lowers fps."))
        self.w_qbias = LabeledSlider("Cleaner frames", "More pixels",
                                     fmt=lambda v: "balanced" if 40 <= v <= 60 else f"{v}")
        g.addRow(self.w_qbias)
        g.addRow(self._hint("Left downscales more so each frame stays crisp and artifact-free; "
                            "right keeps resolution higher and accepts more compression artifacts."))
        self.w_probe = QCheckBox("Smart analysis (measure how hard the content is to compress)")
        self.w_probe.setToolTip("Encodes a few short samples on the GPU when files are added (~1-3 s),\n"
                                "so fast action gets downscaled more than static footage.")
        g.addRow(self.w_probe)

        g = self._group("Accuracy", lay)
        self.w_margin = QDoubleSpinBox()
        self.w_margin.setRange(0, 25)
        self.w_margin.setSuffix(" %")
        self.w_margin.setToolTip("Aim this much below the target to be safe.")
        g.addRow("Safety margin", self.w_margin)
        self.w_attempts = QSpinBox()
        self.w_attempts.setRange(1, 5)
        self.w_attempts.setToolTip("If the result is over the target, re-encode with a corrected bitrate.")
        g.addRow("Max attempts", self.w_attempts)
        self.w_if_smaller = self._combo([("Copy the original", "copy"), ("Skip it", "skip"),
                                         ("Re-encode anyway", "encode")])
        g.addRow("If already small enough", self.w_if_smaller)
        lay.addStretch()
        return w

    def _tab_video(self):
        w, lay = self._page()
        f = self._group("Resolution", lay)
        self.w_res_mode = self._combo([("Auto (calculated)", "auto"), ("Lock to source", "source"),
                                       ("Fixed", "fixed")])
        f.addRow("Mode", self.w_res_mode)
        self.w_fixed_h = QComboBox()
        self.w_fixed_h.setEditable(True)
        for h in (2160, 1440, 1200, 1080, 900, 720, 540, 480, 360):
            self.w_fixed_h.addItem(f"{h}", h)
        f.addRow("Fixed height", self.w_fixed_h)
        self.w_min_h = QSpinBox()
        self.w_min_h.setRange(144, 4320)
        self.w_min_h.setSingleStep(8)
        self.w_min_h.setSuffix("p")
        f.addRow("Never below", self.w_min_h)
        self.w_max_h = QSpinBox()
        self.w_max_h.setRange(144, 4320)
        self.w_max_h.setSingleStep(8)
        self.w_max_h.setSuffix("p")
        f.addRow("Never above", self.w_max_h)
        self.w_snap = QCheckBox("Snap to standard sizes (1080, 900, 720...)")
        f.addRow(self.w_snap)
        f.addRow(self._hint("Heights mean the short side, so a 1080x1920 phone video counts as 1080p."))

        f = self._group("Framerate", lay)
        self.w_fps_mode = self._combo([("Auto (calculated)", "auto"), ("Lock to source", "source"),
                                       ("Fixed", "fixed")])
        f.addRow("Mode", self.w_fps_mode)
        self.w_fixed_fps = QDoubleSpinBox()
        self.w_fixed_fps.setRange(1, 240)
        self.w_fixed_fps.setDecimals(2)
        f.addRow("Fixed fps", self.w_fixed_fps)
        self.w_min_fps = QDoubleSpinBox()
        self.w_min_fps.setRange(1, 240)
        self.w_min_fps.setDecimals(0)
        f.addRow("Never below", self.w_min_fps)
        self.w_max_fps = QDoubleSpinBox()
        self.w_max_fps.setRange(1, 480)
        self.w_max_fps.setDecimals(0)
        f.addRow("Never above", self.w_max_fps)
        self.w_ladder = QLineEdit()
        self.w_ladder.setToolTip("Framerates Auto is allowed to pick from (plus the source fps).")
        f.addRow("Allowed fps", self.w_ladder)
        self.w_clean_fps = QCheckBox("Prefer even divisions (60→30, not 60→48, avoids judder)")
        f.addRow(self.w_clean_fps)
        lay.addStretch()
        return w

    def _tab_encoder(self):
        w, lay = self._page()
        f = self._group("Encoder", lay)
        self.w_encoder = QComboBox()
        for e in E.ENCODERS:
            self.w_encoder.addItem(e.label, e.name)
        f.addRow("Codec", self.w_encoder)
        self.enc_hint = self._hint("")
        f.addRow(self.enc_hint)
        self.w_speed = LabeledSlider("Fastest", "Slowest", 1, 7,
                                     fmt=lambda v: f"{v}" + ("  (recommended)" if v == 4 else ""))
        self.w_speed.setToolTip("NVENC presets p1-p7. Measured on a 1440p60 clip: p4 is ~40% faster than p5 "
                                "with the same quality;\np5-p7 barely improve anything. AV1 loses almost "
                                "nothing even at 2-3.")
        f.addRow("Speed", self.w_speed)

        f = self._group("Rate control", lay)
        self.w_rc = self._combo([("VBR (better quality)", "vbr"), ("CBR (strict size)", "cbr")])
        f.addRow("Mode", self.w_rc)
        self.w_multipass = self._combo([("Off", "disabled"), ("Quarter-res pre-pass", "qres"),
                                        ("Full-res pre-pass", "fullres")])
        self.w_multipass.setToolTip("NVENC's built-in 2-pass. Costs little speed on RTX cards.")
        f.addRow("NVENC multipass", self.w_multipass)
        self.w_lookahead = QSpinBox()
        self.w_lookahead.setRange(0, 60)
        self.w_lookahead.setSuffix(" frames")
        f.addRow("Lookahead", self.w_lookahead)
        self.w_saq = QCheckBox("Spatial AQ (more bits to flat areas, less banding)")
        self.w_taq = QCheckBox("Temporal AQ (better on static parts of the frame)")
        self.w_10bit = QCheckBox("10-bit output (HEVC/AV1 - less banding, slightly less compatible)")
        for c in (self.w_saq, self.w_taq, self.w_10bit):
            f.addRow(c)

        f = self._group("Processing", lay)
        self.w_decoder = self._combo([("Auto", "auto"), ("CPU (all cores)", "cpu"), ("GPU (NVDEC)", "gpu")])
        self.w_decoder.setToolTip("Auto uses CPU decode (usually fastest) and NVDEC for heavy 4K HEVC/AV1.\n"
                                  "Falls back to CPU automatically if the GPU can't decode a file.")
        f.addRow("Decoder", self.w_decoder)
        self.w_scaler = self._combo([("Lanczos (sharpest)", "lanczos"), ("Bicubic", "bicubic"),
                                     ("Bilinear (fastest)", "bilinear")])
        f.addRow("Scaler", self.w_scaler)
        self.w_tonemap = QCheckBox("Convert HDR to SDR (fixes washed-out phone HDR clips)")
        f.addRow(self.w_tonemap)
        lay.addStretch()
        return w

    def _tab_audio(self):
        w, lay = self._page()
        f = self._group("Audio", lay)
        self.w_audio_mode = self._combo([("First track only", "first"), ("Mix all tracks into one", "mix"),
                                         ("Keep all tracks", "all"), ("No audio", "none")])
        self.w_audio_mode.setToolTip("OBS recordings often have separate game / mic tracks - "
                                     "'Mix all' merges them so Discord plays everything.")
        f.addRow("Tracks", self.w_audio_mode)
        self.w_audio_codec = self._combo([("AAC (compatible)", "aac"), ("Opus (better at low bitrate)", "opus"),
                                          ("Copy original", "copy")])
        f.addRow("Codec", self.w_audio_codec)
        self.w_audio_br = QComboBox()
        for b in (32, 48, 64, 96, 128, 160, 192, 256, 320):
            self.w_audio_br.addItem(f"{b} kbps", b)
        f.addRow("Bitrate", self.w_audio_br)
        self.w_audio_auto = QCheckBox("Lower audio bitrate automatically when the budget is tiny")
        self.w_mono = QCheckBox("Downmix to mono")
        f.addRow(self.w_audio_auto)
        f.addRow(self.w_mono)
        lay.addStretch()
        return w

    def _tab_output(self):
        w, lay = self._page()
        f = self._group("Files", lay)
        self.w_same_dir = QCheckBox("Save next to the original")
        f.addRow(self.w_same_dir)
        self.w_out_dir = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_out)
        r = QHBoxLayout()
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(self.w_out_dir, 1)
        r.addWidget(browse)
        self.row_out = QWidget()
        self.row_out.setLayout(r)
        f.addRow("Folder", self.row_out)
        self.w_template = QLineEdit()
        f.addRow("File name", self.w_template)
        f.addRow(self._hint("Tokens: {name} {target} {res} {fps} {codec}"))
        self.w_container = self._combo([("MP4", "mp4"), ("MKV", "mkv")])
        f.addRow("Container", self.w_container)
        self.w_overwrite = QCheckBox("Overwrite existing files (otherwise adds (2), (3)...)")
        self.w_strip = QCheckBox("Strip metadata")
        f.addRow(self.w_overwrite)
        f.addRow(self.w_strip)

        f = self._group("Performance", lay)
        self.w_parallel = QSpinBox()
        self.w_parallel.setRange(1, 6)
        self.w_parallel.setToolTip("Files encoded at the same time. 2 gets ~1.5x more throughput on\n"
                                   "RTX cards; more than 3 rarely helps.")
        f.addRow("Parallel jobs", self.w_parallel)
        self.w_lowprio = QCheckBox("Low process priority (keeps the PC responsive)")
        f.addRow(self.w_lowprio)

        f = self._group("When finished", lay)
        self.w_sound = QCheckBox("Play a sound")
        self.w_open_done = QCheckBox("Open the output folder")
        f.addRow(self.w_sound)
        f.addRow(self.w_open_done)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------ settings binding
    def _bindings(self):
        return [
            (self.w_target_mode, "target_mode"), (self.w_size, "target_size"), (self.w_unit, "size_unit"),
            (self.w_percent, "target_percent"), (self.w_bitrate, "target_bitrate"), (self.w_cq, "cq"),
            (self.w_priority, "priority"), (self.w_qbias, "quality_bias"), (self.w_probe, "smart_probe"),
            (self.w_margin, "safety_margin"), (self.w_attempts, "max_attempts"), (self.w_if_smaller, "if_smaller"),
            (self.w_res_mode, "res_mode"), (self.w_fixed_h, "fixed_height"), (self.w_min_h, "min_height"),
            (self.w_max_h, "max_height"), (self.w_snap, "snap_standard"),
            (self.w_fps_mode, "fps_mode"), (self.w_fixed_fps, "fixed_fps"), (self.w_min_fps, "min_fps"),
            (self.w_max_fps, "max_fps"), (self.w_ladder, "fps_ladder"), (self.w_clean_fps, "prefer_clean_fps"),
            (self.w_encoder, "encoder"), (self.w_speed, "speed"), (self.w_rc, "rate_control"),
            (self.w_multipass, "multipass"), (self.w_lookahead, "lookahead"), (self.w_saq, "spatial_aq"),
            (self.w_taq, "temporal_aq"), (self.w_10bit, "ten_bit"), (self.w_decoder, "decoder"),
            (self.w_scaler, "scaler"), (self.w_tonemap, "tonemap_hdr"),
            (self.w_audio_mode, "audio_mode"), (self.w_audio_codec, "audio_codec"), (self.w_audio_br, "audio_bitrate"),
            (self.w_audio_auto, "audio_auto"), (self.w_mono, "audio_mono"),
            (self.w_out_dir, "output_dir"), (self.w_template, "name_template"), (self.w_container, "container"),
            (self.w_overwrite, "overwrite"), (self.w_strip, "strip_metadata"), (self.w_parallel, "parallel_jobs"),
            (self.w_lowprio, "low_priority"),
        ]

    def _load_widgets(self):
        self._binding = True
        s = self.settings
        for w, attr in self._bindings():
            v = getattr(s, attr)
            if isinstance(w, QComboBox) and w.isEditable():
                w.setCurrentText(str(v))
            elif isinstance(w, QComboBox):
                i = w.findData(v)
                if i < 0 and isinstance(v, (int, float)):
                    w.addItem(f"{v}", v)
                    i = w.count() - 1
                w.setCurrentIndex(max(0, i))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QLineEdit):
                w.setText(str(v))
            else:
                w.setValue(v)
        self.w_same_dir.setChecked(not s.output_dir)
        self.turbo_btn.setChecked(s.turbo)
        self.w_sound.setChecked(self.ui_state.get("play_sound", True))
        self.w_open_done.setChecked(self.ui_state.get("open_folder_when_done", False))
        self._binding = False

        for w, attr in self._bindings():
            if isinstance(w, QComboBox) and w.isEditable():
                w.currentTextChanged.connect(lambda _=None, w=w, a=attr: self._widget_changed(w, a))
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda _=None, w=w, a=attr: self._widget_changed(w, a))
            elif isinstance(w, QCheckBox):
                w.toggled.connect(lambda _=None, w=w, a=attr: self._widget_changed(w, a))
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(lambda _=None, w=w, a=attr: self._widget_changed(w, a))
            else:
                w.valueChanged.connect(lambda _=None, w=w, a=attr: self._widget_changed(w, a))
        self.w_same_dir.toggled.connect(self._same_dir_toggled)
        self.w_sound.toggled.connect(lambda v: self._ui_changed("play_sound", v))
        self.w_open_done.toggled.connect(lambda v: self._ui_changed("open_folder_when_done", v))

    def _widget_changed(self, w, attr):
        if self._binding:
            return
        if isinstance(w, QComboBox) and w.isEditable():
            try:
                v = int(float(w.currentText()))
            except ValueError:
                return
        elif isinstance(w, QComboBox):
            v = w.currentData()
        elif isinstance(w, QCheckBox):
            v = w.isChecked()
        elif isinstance(w, QLineEdit):
            v = w.text()
        else:
            v = w.value()
        setattr(self.settings, attr, v)
        if attr == "smart_probe" and v:
            for job in self.jobs.values():
                if job.info and job.complexity is None and not job.probing_complexity:
                    self._start_complexity(job)
        self._update_visibility()
        self._replan_timer.start()
        self._save_timer.start()

    def _ui_changed(self, key, v):
        self.ui_state[key] = v
        self._save_timer.start()

    def _turbo_toggled(self, on: bool):
        if self._binding:
            return
        self.settings.turbo = on
        self._save_timer.start()

    def _same_dir_toggled(self, same: bool):
        if same:
            self.w_out_dir.setText("")
        elif not self.w_out_dir.text():
            self._browse_out()
            if not self.w_out_dir.text():
                self.w_same_dir.setChecked(True)
        self._update_visibility()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.w_out_dir.text() or os.path.expanduser("~"))
        if d:
            self.w_out_dir.setText(os.path.normpath(d))
            self.w_same_dir.setChecked(False)

    def _update_visibility(self):
        s = self.settings
        mode = s.target_mode

        def show(w, vis):
            f = w.parentWidget().layout()
            if isinstance(f, QFormLayout):
                f.setRowVisible(w, vis)
        show(self.row_size, mode == "size")
        show(self.row_presets, mode == "size")
        show(self.w_percent, mode == "percent")
        show(self.w_bitrate, mode == "bitrate")
        show(self.w_cq, mode == "quality")
        sized = mode in ("size", "percent")
        for w in (self.w_margin, self.w_attempts, self.w_if_smaller):
            w.setEnabled(sized)
        auto = mode != "quality"
        self.w_priority.setEnabled(auto)
        self.w_qbias.setEnabled(auto)
        self.w_fixed_h.setEnabled(s.res_mode == "fixed")
        self.w_min_h.setEnabled(s.res_mode == "auto")
        self.w_max_h.setEnabled(s.res_mode == "auto")
        self.w_fixed_fps.setEnabled(s.fps_mode == "fixed")
        for w in (self.w_min_fps, self.w_max_fps, self.w_ladder, self.w_clean_fps):
            w.setEnabled(s.fps_mode == "auto")
        enc = E.ENCODER_BY_NAME.get(s.encoder)
        nv = enc is not None and enc.family == "nvenc"
        for w in (self.w_multipass, self.w_lookahead, self.w_saq, self.w_taq):
            w.setEnabled(nv)
        self.w_rc.setEnabled(enc is not None and enc.family != "cpu")
        self.w_10bit.setEnabled(enc is not None and enc.ten_bit)
        self.w_decoder.setEnabled(nv)
        self.w_audio_br.setEnabled(s.audio_codec != "copy" or s.audio_mode == "mix")
        self.row_out.setEnabled(bool(s.output_dir) or not self.w_same_dir.isChecked())
        if enc:
            extra = ("  Runs on the GPU - very fast." if enc.family != "cpu"
                     else "  Runs on the CPU - much slower but slightly more efficient.")
            self.enc_hint.setText(CODEC_TIPS[enc.codec].replace("\n", " ") + extra)
            btn = self.codec_btns.get(enc.codec)
            if btn and not btn.isChecked():
                btn.setChecked(True)

    def _pick_codec(self, codec: str):
        """Choose the best available encoder for a codec: GPU (NVENC > QSV > AMF) before CPU,
        keeping the current family when it supports that codec."""
        cur = E.ENCODER_BY_NAME.get(self.settings.encoder)
        avail = self._avail_names or {e.name for e in E.ENCODERS}
        cands = [e for e in E.ENCODERS if e.codec == codec and e.name in avail]
        if not cands:
            return
        same_family = [e for e in cands if cur and e.family == cur.family]
        pick = (same_family or cands)[0]
        self.w_encoder.setCurrentIndex(self.w_encoder.findData(pick.name))

    def _on_encoders_ready(self, encs: list):
        names = {e.name for e in encs}
        self._avail_names = names
        for codec, btn in self.codec_btns.items():
            ok = any(e.codec == codec for e in encs)
            btn.setEnabled(ok)
            if not ok:
                btn.setToolTip("No encoder for this codec is available on this PC.")
        model = self.w_encoder.model()
        for i in range(self.w_encoder.count()):
            name = self.w_encoder.itemData(i)
            item = model.item(i)
            ok = name in names
            item.setEnabled(ok)
            if not ok:
                item.setText(E.ENCODER_BY_NAME[name].label + "  (not available)")
        if self.settings.encoder not in names and encs:
            self.w_encoder.setCurrentIndex(self.w_encoder.findData(encs[0].name))
        # kick off complexity analysis that waited for encoder detection
        for job in self.jobs.values():
            if job.info and job.complexity is None and self.settings.smart_probe and not job.probing_complexity:
                self._start_complexity(job)

    # ------------------------------------------------------------------ config
    def _load_config(self):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        self.settings = E.Settings.from_dict(data.get("settings", {}))
        if data.get("version", 1) < 2 and self.settings.speed == 5:
            self.settings.speed = 4  # old default; p5 turned out slower with no quality gain
        if data.get("version", 1) < 3:  # old defaults that benchmarks showed cost speed for no quality
            if self.settings.lookahead == 20:
                self.settings.lookahead = 0
            if self.settings.multipass == "qres":
                self.settings.multipass = "disabled"
        if data.get("version", 1) < 4 and self.settings.encoder == "h264_nvenc":
            self.settings.encoder = "hevc_nvenc"  # new default
        self.ui_state.update(data.get("ui", {}))
        E.set_calibration(data.get("calibration", {}))
        geo = data.get("ui", {}).get("geometry")
        if geo and len(geo) == 4:
            self.setGeometry(*geo)

    def _save_config(self):
        g = self.geometry()
        self.ui_state["geometry"] = [g.x(), g.y(), g.width(), g.height()]
        self.ui_state["splitter"] = self.splitter.sizes()
        data = {"version": 4, "settings": self.settings.to_dict(), "ui": self.ui_state, "calibration": E.calibration()}
        try:
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, CONFIG_PATH)
        except OSError:
            pass

    # ------------------------------------------------------------------ adding files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def _add_files_dialog(self):
        exts = " ".join(f"*{x}" for x in sorted(E.VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add videos", self.ui_state.get("last_dir", ""),
                                                f"Videos ({exts});;All files (*)")
        if files:
            self.ui_state["last_dir"] = os.path.dirname(files[0])
            self.add_paths(files)

    def _add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder", self.ui_state.get("last_dir", ""))
        if d:
            self.ui_state["last_dir"] = d
            self.add_paths([d])

    def add_paths(self, paths: list[str]):
        files = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, names in os.walk(p):
                    for n in sorted(names):
                        if os.path.splitext(n)[1].lower() in E.VIDEO_EXTS and ".part." not in n:
                            files.append(os.path.join(root, n))
            elif os.path.isfile(p):
                files.append(p)
        existing = {os.path.normcase(j.path) for j in self.jobs.values()}
        added = 0
        for f in files:
            f = os.path.normpath(f)
            if os.path.normcase(f) in existing:
                continue
            existing.add(os.path.normcase(f))
            job = Job(f)
            self.jobs[job.id] = job
            self._add_row(job)
            self.analysis_pool.submit(self._analyse, job)
            added += 1
        self.table.update_empty()
        if added:
            self._set_status(f"Added {added} file{'s' if added != 1 else ''}. Analysing...")

    def _analyse(self, job: Job):
        try:
            job.info = E.probe(job.path)
            job.status = "ready" if job.status == "analysing" else job.status
        except Exception as ex:  # noqa: BLE001
            job.status, job.error = "failed", f"Can't read file: {ex}"
            job.analysed.set()
            self.bus.job_changed.emit(job.id)
            return
        self.bus.job_changed.emit(job.id)
        if (self.settings.smart_probe and E._available_cache is not None
                and job.complexity is None and not job.probing_complexity):
            self._complexity(job)
        job.analysed.set()

    def _start_complexity(self, job: Job):
        job.probing_complexity = True
        self.analysis_pool.submit(self._complexity, job)

    def _complexity(self, job: Job):
        job.probing_complexity = True
        self.bus.job_changed.emit(job.id)
        try:
            job.complexity = E.analyze_complexity(job.info, job.trim)
        finally:
            job.probing_complexity = False
            self.bus.job_changed.emit(job.id)

    # ------------------------------------------------------------------ table rendering
    def _row_of(self, job_id: int) -> int:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_FILE)
            if it and it.data(Qt.UserRole) == job_id:
                return r
        return -1

    def _add_row(self, job: Job):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(len(HEADERS)):
            it = QTableWidgetItem("")
            if c == COL_FILE:
                it.setData(Qt.UserRole, job.id)
            self.table.setItem(r, c, it)
        self._refresh_job(job.id)

    def _refresh_job(self, job_id: int):
        job = self.jobs.get(job_id)
        r = self._row_of(job_id)
        if job is None or r < 0:
            return
        s = self.run_settings if (self.running and job in self.run_jobs) else self.settings
        info = job.info
        it = self.table.item(r, COL_FILE)
        it.setText(job.name)
        it.setToolTip(job.path)
        src = self.table.item(r, COL_SRC)
        if info:
            src.setText(info.describe())
            src.setToolTip(f"{info.width}x{info.height} @ {info.fps:.3f} fps\n"
                           f"Codec: {info.vcodec} ({info.pix_fmt}{', HDR' if info.hdr else ''})\n"
                           f"Audio tracks: {len(info.audio)}\n"
                           f"Content complexity: {complexity_label(job.complexity)}")
        else:
            src.setText("reading..." if job.status == "analysing" else "-")

        plan_it = self.table.item(r, COL_PLAN)
        if info and job.status not in ("failed",) or (info and job.result):
            if job.status in ("ready", "queued", "analysing") or job.plan is None:
                try:
                    job.plan = E.make_plan(info, s, job.complexity if s.smart_probe else None, job.trim)
                except Exception as ex:  # noqa: BLE001
                    job.plan = None
                    plan_it.setText(f"error: {ex}")
            p = job.plan
            if p:
                txt = p.summary()
                if p.quality and p.action == "encode":
                    txt += f" · {p.quality}"
                if job.probing_complexity:
                    txt += "  (analysing...)"
                plan_it.setText(txt)
                plan_it.setForeground(QColor(QUALITY_COLORS.get(p.quality, "#c9cad6")))
                tip = [f"Output: {p.width}x{p.height} @ {p.fps:.3f} fps"]
                if p.video_kbps is not None:
                    tip.append(f"Video: {E.fmt_kbps(p.video_kbps)}")
                if p.audio_streams:
                    tip.append(f"Audio: {p.audio_kbps} kbps x{p.audio_streams}")
                if p.est_bytes():
                    tip.append(f"Expected size: ≤ {E.fmt_size(p.est_bytes())}")
                tip.append(f"Content complexity: {complexity_label(job.complexity)}")
                if p.est_qp is not None:
                    tip.append(f"Predicted quality: {p.quality} (≈QP {p.est_qp:.0f})")
                tip += [f"Note: {n}" for n in p.notes]
                plan_it.setToolTip("\n".join(tip))
        elif job.status == "failed" and not info:
            plan_it.setText("-")

        trim = self.table.item(r, COL_TRIM)
        if job.trim != (None, None):
            a = E.fmt_dur(job.trim[0] or 0)
            b = E.fmt_dur(job.trim[1]) if job.trim[1] else "end"
            trim.setText(f"{a}-{b}")
        else:
            trim.setText("-")

        self._refresh_progress(job_id)

        res = self.table.item(r, COL_RESULT)
        if job.status == "done" and job.result:
            rr = job.result
            if job.plan and job.plan.action == "skip":
                res.setText("skipped")
            elif job.plan and job.plan.action == "copy":
                res.setText(f"{E.fmt_size(rr.size)} (copied)")
            else:
                res.setText(f"{E.fmt_size(rr.size)} · {rr.speed:.1f}x")
            tip = [f"Output: {rr.output}"] if rr.output else []
            if info and rr.size:
                tip.append(f"{E.fmt_size(info.size)} → {E.fmt_size(rr.size)} ({rr.size / info.size * 100:.1f}%)")
            if rr.attempts > 1:
                tip.append(f"Took {rr.attempts} attempts to fit the target")
            if rr.over_target:
                tip.append("⚠ Still over the target size - lower the target or raise attempts/margin")
            if rr.gpu_fallback:
                tip.append("GPU decode failed for this file, used CPU decode")
            tip.append(f"Time: {rr.elapsed:.1f}s")
            res.setToolTip("\n".join(tip))
            res.setForeground(QColor("#fb923c" if rr.over_target else "#c9cad6"))
        elif job.status == "failed":
            res.setText("error")
            res.setToolTip(job.error)
            res.setForeground(QColor("#f87171"))
        else:
            res.setText("")
            res.setToolTip("")

    def _refresh_progress(self, job_id: int):
        job = self.jobs.get(job_id)
        r = self._row_of(job_id)
        if job is None or r < 0:
            return
        it = self.table.item(r, COL_PROG)
        st = job.status
        if st == "encoding":
            sp = f" · {job.speed:.1f}x" if job.speed else ""
            data = (job.progress, f"{job.stage} {job.progress * 100:.0f}%{sp}", ACCENT)
        elif st == "done":
            data = (1.0, "Done", "#2f9e5b")
        elif st == "failed":
            data = (1.0, "Failed", "#b33a3a")
        elif st == "cancelled":
            data = (None, "Cancelled", "")
        elif st == "analysing" or job.probing_complexity:
            data = (None, "Analysing...", "")
        elif st == "queued":
            data = (None, "Queued", "")
        elif False:
            data = (None, "Analysing...", "")
        else:
            data = (None, "Ready", "")
        it.setData(Qt.UserRole, data)
        it.setToolTip(job.error if st == "failed" else "")
        self.table.viewport().update(self.table.visualItemRect(it))

    def _replan_all(self):
        if self.running:
            for j in self.jobs.values():
                if j not in self.run_jobs:
                    self._refresh_job(j.id)
        else:
            for j in self.jobs.values():
                self._refresh_job(j.id)

    # ------------------------------------------------------------------ context menu
    def _selected_jobs(self) -> list[Job]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        out = []
        for r in rows:
            jid = self.table.item(r, COL_FILE).data(Qt.UserRole)
            if jid in self.jobs:
                out.append(self.jobs[jid])
        return out

    def _context_menu(self, pos):
        jobs = self._selected_jobs()
        if not jobs:
            return
        m = QMenu(self)
        one = jobs[0] if len(jobs) == 1 else None
        done = [j for j in jobs if j.status == "done" and j.result and j.result.output]
        if done:
            m.addAction("Copy output file (paste into Discord)", lambda: self._copy_files(done))
            m.addAction("Show output in Explorer", lambda: reveal(done[0].result.output))
            m.addAction("Play output", lambda: os.startfile(done[0].result.output))
            m.addSeparator()
        if one:
            m.addAction("Play original", lambda: os.startfile(one.path))
            m.addAction("Show original in Explorer", lambda: reveal(one.path))
        idle = [j for j in jobs if j.status not in ("encoding", "queued")]
        if one and one.info and one in idle:
            m.addSeparator()
            m.addAction("Set trim...", lambda: self._set_trim(one))
            if one.trim != (None, None):
                m.addAction("Clear trim", lambda: self._apply_trim(one, (None, None)))
        if idle:
            m.addSeparator()
            redo = [j for j in idle if j.status in ("done", "failed", "cancelled")]
            if redo:
                m.addAction("Reset to ready (compress again)", lambda: self._reset(redo))
            m.addAction("Remove", self._remove_selected)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _double_clicked(self, index):
        jid = self.table.item(index.row(), COL_FILE).data(Qt.UserRole)
        job = self.jobs.get(jid)
        if not job:
            return
        if job.status == "done" and job.result and job.result.output:
            reveal(job.result.output)
        elif index.column() == COL_TRIM and job.info and job.status not in ("encoding", "queued"):
            self._set_trim(job)

    def _copy_files(self, jobs: list[Job]):
        md = QMimeData()
        md.setUrls([QUrl.fromLocalFile(j.result.output) for j in jobs])
        QApplication.clipboard().setMimeData(md)
        self._set_status(f"Copied {len(jobs)} file{'s' if len(jobs) > 1 else ''} to the clipboard - "
                         "paste with Ctrl+V in Discord.")

    def _set_trim(self, job: Job):
        d = TrimDialog(self, job)
        if d.exec():
            self._apply_trim(job, d.value)

    def _apply_trim(self, job: Job, trim):
        job.trim = trim
        job.complexity = None
        if job.status in ("done", "failed", "cancelled"):
            job.status, job.result, job.error, job.progress = "ready", None, "", 0.0
        if self.settings.smart_probe:
            self._start_complexity(job)
        self._refresh_job(job.id)

    def _reset(self, jobs: list[Job]):
        for j in jobs:
            if j.info:
                j.status, j.result, j.error, j.progress, j.plan = "ready", None, "", 0.0, None
            self._refresh_job(j.id)

    def _remove_selected(self):
        for j in self._selected_jobs():
            if j.status in ("encoding", "queued"):
                continue
            r = self._row_of(j.id)
            if r >= 0:
                self.table.removeRow(r)
            self.jobs.pop(j.id, None)
        self.table.update_empty()

    def _clear_finished(self):
        for j in list(self.jobs.values()):
            if j.status in ("done", "failed", "cancelled"):
                r = self._row_of(j.id)
                if r >= 0:
                    self.table.removeRow(r)
                self.jobs.pop(j.id, None)
        self.table.update_empty()

    def _clear_all(self):
        if self.running:
            self._clear_finished()
            return
        self.table.setRowCount(0)
        self.jobs.clear()
        self.table.update_empty()
        self._set_status("Queue cleared.")

    # ------------------------------------------------------------------ running
    def start(self):
        if self.running:
            return
        todo = [j for j in self.jobs.values() if j.status in ("ready", "analysing", "cancelled")]
        if not todo:
            if any(j.status == "done" for j in self.jobs.values()):
                self._set_status("Everything is done. Right-click → Reset to compress again.")
            else:
                self._set_status("Nothing to compress - add some videos first.")
            return
        s = self.settings
        if s.output_dir and not os.path.isdir(s.output_dir):
            try:
                os.makedirs(s.output_dir, exist_ok=True)
            except OSError as ex:
                QMessageBox.warning(self, APP_NAME, f"Can't create output folder:\n{ex}")
                return
        self._save_config()
        self.running = True
        self.run_settings = E.Settings.from_dict(s.to_dict())
        self.run_jobs = todo
        self.run_started = time.monotonic()
        self._reserved = set()
        for j in todo:
            j.status, j.progress, j.speed, j.error, j.result = "queued", 0.0, None, "", None
            j.token = E.CancelToken()
            self._refresh_job(j.id)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._overall_timer.start()
        self._enc_busy = None

        def check_load():
            time.sleep(0.3)  # before our own sessions start
            self._enc_busy = E.nvenc_load()
        threading.Thread(target=check_load, daemon=True).start()
        threading.Thread(target=self._run_all, args=(todo, self.run_settings), daemon=True).start()

    def _run_all(self, jobs: list[Job], s: E.Settings):
        with ThreadPoolExecutor(max(1, s.parallel_jobs)) as ex:
            list(ex.map(lambda j: self._run_one(j, s), jobs))
        self.bus.run_finished.emit()

    def _run_one(self, job: Job, s: E.Settings):
        token = job.token
        if token is None or token.cancelled:
            return
        job.analysed.wait()
        if job.info is None:
            job.status = "failed"
            self.bus.job_changed.emit(job.id)
            return
        while job.probing_complexity and not token.cancelled:
            time.sleep(0.1)
        if s.smart_probe and job.complexity is None and not token.cancelled:
            job.stage = "Analysing"
            job.probing_complexity = True
            self.bus.job_changed.emit(job.id)
            try:
                job.complexity = E.analyze_complexity(job.info, job.trim)
            finally:
                job.probing_complexity = False
        if token.cancelled:
            return
        try:
            job.plan = E.make_plan(job.info, s, job.complexity if s.smart_probe else None, job.trim)
            with self._reserve_lock:
                out = E.output_path_for(job.info, job.plan, s, self._reserved)
            job.output = out
            job.status, job.started, job.stage = "encoding", time.monotonic(), "Encoding"
            self.bus.job_changed.emit(job.id)
            last = [0.0]

            def cb(frac, speed, stage):
                job.progress, job.stage = frac, stage
                if speed:
                    job.speed = speed
                now = time.monotonic()
                if now - last[0] > 0.2:
                    last[0] = now
                    self.bus.job_progress.emit(job.id)

            job.result = E.encode(job.info, job.plan, s, out, job.trim, cb, token)
            job.status, job.progress = "done", 1.0
        except E.Cancelled:
            job.status = "cancelled"
        except Exception as ex:  # noqa: BLE001
            job.status, job.error = "failed", str(ex)
        self.bus.job_changed.emit(job.id)

    def stop(self):
        for j in self.run_jobs:
            if j.token:
                j.token.cancel()
            if j.status == "queued":
                j.status = "cancelled"
                self._refresh_job(j.id)
        self._set_status("Stopping...")

    def _on_run_finished(self):
        self.running = False
        self._overall_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        jobs = self.run_jobs
        done = [j for j in jobs if j.status == "done"]
        failed = [j for j in jobs if j.status == "failed"]
        cancelled = [j for j in jobs if j.status == "cancelled"]
        src = sum(j.info.size for j in done if j.info)
        out = sum(j.result.size for j in done if j.result and j.result.output)
        el = time.monotonic() - self.run_started
        msg = f"Finished {len(done)} file{'s' if len(done) != 1 else ''} in {E.fmt_dur(el)}"
        if done:
            msg += f"  ·  {E.fmt_size(src)} → {E.fmt_size(out)}"
        if failed:
            msg += f"  ·  {len(failed)} failed (hover the row for details)"
        if cancelled:
            msg += f"  ·  {len(cancelled)} cancelled"
        self._set_status(msg)
        self.overall.setValue(1000 if done and not failed and not cancelled else self.overall.value())
        for j in jobs:
            self._refresh_job(j.id)
        self._save_config()  # persists learned rate-control calibration
        if done and not cancelled:
            if self.ui_state.get("play_sound", True):
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
                except Exception:  # noqa: BLE001
                    QApplication.beep()
            if self.ui_state.get("open_folder_when_done") and done[0].result and done[0].result.output:
                reveal(done[0].result.output)

    def _update_overall(self):
        jobs = self.run_jobs
        if not jobs:
            return
        total = sum((j.plan.duration if j.plan else (j.info.duration if j.info else 1)) for j in jobs)
        prog = 0.0
        for j in jobs:
            d = j.plan.duration if j.plan else (j.info.duration if j.info else 1)
            if j.status in ("done", "failed", "cancelled"):
                prog += d
            elif j.status == "encoding":
                prog += d * j.progress
        frac = prog / total if total else 0
        self.overall.setValue(int(frac * 1000))
        el = time.monotonic() - self.run_started
        active = [j for j in jobs if j.status == "encoding"]
        n_done = sum(1 for j in jobs if j.status in ("done", "failed", "cancelled"))
        msg = f"Compressing {n_done + len(active)}/{len(jobs)}"
        speed = sum(j.speed or 0 for j in active)
        if speed:
            msg += f"  ·  {speed:.1f}x realtime"
        if frac > 0.03:
            msg += f"  ·  ~{E.fmt_dur(el / frac - el)} left"
        busy = getattr(self, "_enc_busy", None)
        if busy and busy[0] > 0 and busy[1] >= 10:
            msg += (f"  ·  ⚠ another app (OBS replay buffer?) is using {busy[1]}% of the GPU encoder "
                    "- stop it for faster compression")
        self._set_status(msg)

    def _set_status(self, text: str):
        self.status.setText(text)

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, APP_NAME, "Compression is running. Stop and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.stop()
        self._save_config()
        self.analysis_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(e)


# --------------------------------------------------------------------------- helpers

def reveal(path: str):
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ACCENT))
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setBrush(QColor("white"))
    p.drawPolygon(QPolygon([QPoint(24, 18), QPoint(24, 46), QPoint(46, 32)]))
    p.end()
    return QIcon(pm)


def dark_title_bar(widget: QWidget):
    if os.name != "nt":
        return
    try:
        hwnd = int(widget.winId())
        val = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:  # noqa: BLE001
        pass


STYLE = f"""
* {{ font-family: 'Segoe UI Variable Text', 'Segoe UI', sans-serif; font-size: 10pt; color: #e6e6ee; }}
QMainWindow, QWidget {{ background: #16171d; }}
QLabel#title {{ font-size: 14pt; font-weight: 600; }}
QLabel#hint {{ color: #8b8d9c; font-size: 9pt; }}
QLabel#sliderValue {{ color: {ACCENT}; font-weight: 600; font-size: 9pt; }}
QLabel#emptyHint {{ color: #5d6070; font-size: 13pt; background: transparent; }}
QGroupBox {{ border: 1px solid #262833; border-radius: 10px; margin-top: 14px; padding: 12px 10px 10px 10px;
             background: #1b1c23; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #b7b8c6; font-weight: 600; }}
QGroupBox QWidget {{ background: transparent; }}
QPushButton {{ background: #262833; border: 1px solid #30323f; border-radius: 7px; padding: 6px 14px; }}
QPushButton:hover {{ background: #2f3140; }}
QPushButton:pressed {{ background: #22242e; }}
QPushButton:disabled {{ color: #5d6070; }}
QPushButton#primary {{ background: {ACCENT}; border: none; font-weight: 600; padding: 9px 22px; }}
QPushButton#primary:hover {{ background: #7f70ff; }}
QPushButton#primary:disabled {{ background: #3b3570; color: #9d98c8; }}
QPushButton#segL, QPushButton#segM, QPushButton#segR {{ padding: 9px 12px; border-radius: 0; }}
QPushButton#segL {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton#segR {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
QPushButton#segM {{ border-left: none; border-right: none; }}
QPushButton#segL:checked, QPushButton#segM:checked, QPushButton#segR:checked {{
    background: #2d2a55; border-color: {ACCENT}; color: #ffffff; font-weight: 600; }}
QPushButton#turbo {{ padding: 9px 16px; }}
QPushButton#turbo:checked {{ background: #3a3217; border: 1px solid #facc15; color: #facc15; font-weight: 600; }}
QPushButton#chip {{ padding: 4px 8px; border-radius: 12px; min-width: 30px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: #22232c; border: 1px solid #30323f;
    border-radius: 6px; padding: 5px 8px; min-height: 20px; selection-background-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{ color: #5d6070; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: #454859; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/down.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: #22232c; border: 1px solid #30323f; selection-background-color: {ACCENT}; outline: 0; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox:disabled {{ color: #5d6070; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #454859; background: #22232c; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({ASSETS}/check.svg); }}
QCheckBox::indicator:checked:disabled {{ background: #3b3f4c; border-color: #3b3f4c; }}
QSlider::groove:horizontal {{ height: 4px; background: #30323f; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #f1f1f6; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider::sub-page:horizontal:disabled {{ background: #3b3f4c; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 12px; color: #8b8d9c; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: #f1f1f6; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: #d0d1dc; }}
QTableWidget {{ background: #1b1c23; alternate-background-color: #1f2028; border: 1px solid #262833;
    border-radius: 10px; gridline-color: transparent; selection-background-color: #2d2a55; outline: 0; }}
QTableWidget::item {{ padding: 0 8px; border: none; }}
QTableWidget::item:selected {{ background: #2d2a55; color: #ffffff; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: #1b1c23; color: #8b8d9c; border: none; border-bottom: 1px solid #262833;
    padding: 8px; font-weight: 600; }}
QProgressBar {{ background: #262833; border: none; border-radius: 4px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QScrollArea {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #30323f; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #30323f; border-radius: 4px; min-width: 30px; }}
QSplitter::handle {{ background: transparent; width: 10px; }}
QMenu {{ background: #22232c; border: 1px solid #30323f; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: #30323f; margin: 4px 6px; }}
QToolTip {{ background: #22232c; color: #e6e6ee; border: 1px solid #454859; padding: 6px; }}
"""


def main(argv: list[str] | None = None):
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("kanibals.videocompressor")
        except Exception:  # noqa: BLE001
            pass
    app = QApplication(sys.argv if argv is None else argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#16171d"))
    pal.setColor(QPalette.Base, QColor("#1b1c23"))
    pal.setColor(QPalette.Text, QColor("#e6e6ee"))
    pal.setColor(QPalette.Highlight, QColor("#2d2a55"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    dark_title_bar(win)
    win.show()
    files = [a for a in (sys.argv[1:] if argv is None else argv[1:]) if os.path.exists(a)]
    if files:
        win.add_paths(files)
    sys.exit(app.exec())
