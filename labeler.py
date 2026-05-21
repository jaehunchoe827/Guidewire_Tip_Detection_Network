"""Image labeler GUI: click one pixel per frame, see a 5x magnifier, auto-save.

Layout:
- Top-left: current image at its original resolution.
- Top-right: 1000x1000 magnifier showing the 200x200 region around the cursor
  (5x nearest-neighbor upscale) with a green crosshair at the exact cursor
  pixel. The selection circle is also drawn in the magnifier when visible.
- Below the magnifier: text showing the selected pixel coordinates.
- Bottom: video selector, image-name jump field, and a counter.

Labels are written to ``datasets/<video>/Labels/<name>.txt`` as a single line
``1 x y`` whenever the user navigates away from an image (or closes the window)
while a pixel is selected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# opencv-python ships its own Qt5 platform plugins under cv2/qt/plugins and,
# on import, unconditionally points QT_QPA_PLATFORM_PLUGIN_PATH at them
# (see cv2/config-3.py). Those plugins are built against a different Qt than
# the system PyQt5 (Qt 5.15.3 here), which is why PyQt5 then fails with
# "Could not load the Qt platform plugin 'xcb'" / "QObject::moveToThread".
# Import cv2 first, then redirect the Qt plugin path back to the system /
# PyQt5 plugins before any PyQt5 module is loaded.
import cv2  # noqa: E402


def _restore_system_qt_plugin_path() -> None:
    candidates: list[str] = []
    try:
        import PyQt5 as _pyqt5  # type: ignore

        pyqt_dir = os.path.dirname(_pyqt5.__file__)
        candidates += [
            os.path.join(pyqt_dir, "Qt5", "plugins"),
            os.path.join(pyqt_dir, "Qt", "plugins"),
        ]
    except Exception:
        pass
    candidates += [
        "/usr/lib/x86_64-linux-gnu/qt5/plugins",
        "/usr/lib/qt5/plugins",
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "platforms")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = path
            os.environ.pop("QT_QPA_FONTDIR", None)
            return


_restore_system_qt_plugin_path()

from PyQt5.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt5.QtGui import (  # noqa: E402
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets" / "guidewire"

MAG_SCALE = 5
MAG_RADIUS = 100  # source crop half-size -> 200x200 region around cursor
MAG_SIZE = MAG_RADIUS * 2 * MAG_SCALE  # 1000

CIRCLE_RADIUS = 5
GREEN = QColor(0, 255, 0)


def images_dir(video: str) -> Path:
    return DATASETS_DIR / video / "Images"


def labels_dir(video: str) -> Path:
    return DATASETS_DIR / video / "Labels"


def label_file(video: str, name: int) -> Path:
    return labels_dir(video) / f"{name}.txt"


def list_videos(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "Images").is_dir()
    )


def list_image_names(video: str) -> list[int]:
    d = images_dir(video)
    if not d.is_dir():
        return []
    names: list[int] = []
    for p in d.iterdir():
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        try:
            names.append(int(p.stem))
        except ValueError:
            continue
    names.sort()
    return names


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert an HxWx3 BGR uint8 image into a self-owning QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # Detach from the numpy buffer before it can be freed.
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QLabel):
    """QLabel that paints an image plus an optional green selection circle."""

    clicked = pyqtSignal(int, int)
    cursorMoved = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pix: Optional[QPixmap] = None
        self._point: Optional[tuple[int, int]] = None

    def set_image(self, pix: Optional[QPixmap]) -> None:
        self._pix = pix
        if pix is None or pix.isNull():
            self.setFixedSize(1, 1)
        else:
            self.setFixedSize(pix.width(), pix.height())
        self.update()

    def set_point(self, point: Optional[tuple[int, int]]) -> None:
        self._point = point
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401 (Qt override)
        painter = QPainter(self)
        if self._pix is not None and not self._pix.isNull():
            painter.drawPixmap(0, 0, self._pix)
        if self._point is not None:
            x, y = self._point
            painter.setPen(QPen(GREEN, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2,
            )
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._pix is None or self._pix.isNull():
            return
        x = int(event.pos().x())
        y = int(event.pos().y())
        if 0 <= x < self._pix.width() and 0 <= y < self._pix.height():
            self.clicked.emit(x, y)
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:
        if self._pix is None or self._pix.isNull():
            return
        x = int(event.pos().x())
        y = int(event.pos().y())
        if 0 <= x < self._pix.width() and 0 <= y < self._pix.height():
            self.cursorMoved.emit(x, y)


class LabelerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Labeler")

        self.videos: list[str] = list_videos(DATASETS_DIR)
        if not self.videos:
            QMessageBox.critical(
                self, "No datasets",
                f"No videos with an Images/ folder were found under\n{DATASETS_DIR}",
            )
            raise SystemExit(1)

        self.video: str = self.videos[0]
        self.image_names: list[int] = []
        self.idx: int = 0
        self.bgr: Optional[np.ndarray] = None
        self.point: Optional[tuple[int, int]] = None
        self.cursor: Optional[tuple[int, int]] = None

        self._build_ui()
        self._install_shortcuts()
        self._populate_videos()
        self._load_video(self.video)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._on_click)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)

        self.magnifier = QLabel()
        self.magnifier.setFixedSize(MAG_SIZE, MAG_SIZE)
        self.magnifier.setStyleSheet("background-color: black;")
        self.magnifier.setAlignment(Qt.AlignCenter)

        self.coord_label = QLabel("(none)")
        f = self.coord_label.font()
        f.setPointSize(14)
        f.setBold(True)
        self.coord_label.setFont(f)
        self.coord_label.setAlignment(Qt.AlignCenter)

        right_col = QVBoxLayout()
        right_col.addWidget(self.magnifier)
        right_col.addWidget(self.coord_label)
        right_col.addStretch(1)

        top_row = QHBoxLayout()
        top_row.addWidget(self.canvas, 0, Qt.AlignTop | Qt.AlignLeft)
        top_row.addLayout(right_col)
        top_row.addStretch(1)

        self.video_combo = QComboBox()
        self.video_combo.currentTextChanged.connect(self._on_video_changed)

        self.image_edit = QLineEdit()
        self.image_edit.setFixedWidth(120)
        self.image_edit.setPlaceholderText("frame #")
        self.image_edit.returnPressed.connect(self._on_image_edit_go)

        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self._on_image_edit_go)

        self.counter_label = QLabel("0 / 0")

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Video:"))
        bottom_row.addWidget(self.video_combo)
        bottom_row.addSpacing(20)
        bottom_row.addWidget(QLabel("Image:"))
        bottom_row.addWidget(self.image_edit)
        bottom_row.addWidget(self.go_button)
        bottom_row.addSpacing(20)
        bottom_row.addWidget(self.counter_label)
        bottom_row.addStretch(1)

        root = QVBoxLayout()
        root.addLayout(top_row, 1)
        root.addLayout(bottom_row)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _install_shortcuts(self) -> None:
        for key in (Qt.Key_A, Qt.Key_Left):
            QShortcut(QKeySequence(key), self, activated=self._go_prev)
        for key in (Qt.Key_D, Qt.Key_Right):
            QShortcut(QKeySequence(key), self, activated=self._go_next)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._clear_point)
        QShortcut(QKeySequence(Qt.Key_Q), self, activated=self.close)

    def _populate_videos(self) -> None:
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for v in self.videos:
            self.video_combo.addItem(v)
        self.video_combo.setCurrentText(self.video)
        self.video_combo.blockSignals(False)

    # ------------------------------------------------------------ Video/image

    def _on_video_changed(self, new_video: str) -> None:
        if not new_video or new_video == self.video:
            return
        self._save_current_label()
        self._load_video(new_video)
        self.canvas.setFocus()

    def _load_video(self, video: str) -> None:
        self.video = video
        self.image_names = list_image_names(video)
        self.idx = 0
        if not self.image_names:
            self.bgr = None
            self.point = None
            self.cursor = None
            self.canvas.set_image(None)
            self.canvas.set_point(None)
            self.magnifier.setPixmap(QPixmap())
            self._update_counter()
            self._update_coord_text()
            return
        self._load_image(self.image_names[self.idx])

    def _load_image(self, name: int) -> None:
        path = images_dir(self.video) / f"{name}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Load error", f"Could not read {path}")
            return
        self.bgr = bgr
        self.point = self._load_label(name)
        self.canvas.set_image(bgr_to_qpixmap(bgr))
        self.canvas.set_point(self.point)
        self._update_counter()
        self._update_coord_text()
        self._refresh_magnifier()

    def _load_label(self, name: int) -> Optional[tuple[int, int]]:
        path = label_file(self.video, name)
        if not path.is_file():
            return None
        try:
            text = path.read_text().strip()
            if not text:
                return None
            tokens = text.splitlines()[0].split()
            if len(tokens) < 3:
                return None
            return int(tokens[1]), int(tokens[2])
        except (OSError, ValueError):
            return None

    def _save_current_label(self) -> None:
        if not self.image_names:
            return
        name = self.image_names[self.idx]
        path = label_file(self.video, name)
        if self.point is None:
            # Cleared selection: ensure no stale label remains on disk.
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        labels_dir(self.video).mkdir(parents=True, exist_ok=True)
        x, y = self.point
        with open(path, "w") as f:
            f.write(f"1 {x} {y}\n")

    # ---------------------------------------------------------------- Events

    def _on_click(self, x: int, y: int) -> None:
        self.point = (x, y)
        self.canvas.set_point(self.point)
        self._update_coord_text()
        self._refresh_magnifier()

    def _on_cursor_moved(self, x: int, y: int) -> None:
        self.cursor = (x, y)
        self._refresh_magnifier()

    def _clear_point(self) -> None:
        if self.point is None:
            return
        self.point = None
        self.canvas.set_point(None)
        self._update_coord_text()
        self._refresh_magnifier()

    def _go_prev(self) -> None:
        if not self.image_names or self.idx <= 0:
            return
        self._save_current_label()
        self.idx -= 1
        self._load_image(self.image_names[self.idx])

    def _go_next(self) -> None:
        if not self.image_names or self.idx >= len(self.image_names) - 1:
            return
        self._save_current_label()
        self.idx += 1
        self._load_image(self.image_names[self.idx])

    def _on_image_edit_go(self) -> None:
        text = self.image_edit.text().strip()
        self.image_edit.clear()
        self.canvas.setFocus()
        if not text:
            return
        # Allow either "5" or "5.jpg".
        if text.lower().endswith(".jpg"):
            text = text[:-4]
        try:
            target = int(text)
        except ValueError:
            return
        if target not in self.image_names:
            return
        self._save_current_label()
        self.idx = self.image_names.index(target)
        self._load_image(self.image_names[self.idx])

    # ----------------------------------------------------------- UI updates

    def _update_counter(self) -> None:
        if not self.image_names:
            self.counter_label.setText("0 / 0")
            return
        name = self.image_names[self.idx]
        self.counter_label.setText(
            f"#{name}   ({self.idx + 1} / {len(self.image_names)})"
        )

    def _update_coord_text(self) -> None:
        if self.point is None:
            self.coord_label.setText("(none)")
        else:
            x, y = self.point
            self.coord_label.setText(f"X: {x}   Y: {y}")

    def _refresh_magnifier(self) -> None:
        if self.bgr is None or self.cursor is None:
            self.magnifier.setPixmap(QPixmap())
            return

        cx, cy = self.cursor
        h, w = self.bgr.shape[:2]
        side = MAG_RADIUS * 2
        x0 = cx - MAG_RADIUS
        y0 = cy - MAG_RADIUS

        crop = np.zeros((side, side, 3), dtype=np.uint8)
        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(w, x0 + side)
        sy1 = min(h, y0 + side)
        if sx1 > sx0 and sy1 > sy0:
            dx = sx0 - x0
            dy = sy0 - y0
            crop[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0)] = \
                self.bgr[sy0:sy1, sx0:sx1]

        mag = cv2.resize(crop, (MAG_SIZE, MAG_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        pix = bgr_to_qpixmap(mag)

        painter = QPainter(pix)
        try:
            # Center of the magnified pixel that the cursor is on.
            center = MAG_RADIUS * MAG_SCALE + MAG_SCALE // 2
            painter.setPen(QPen(GREEN, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(center - 10, center, center + 10, center)
            painter.drawLine(center, center - 10, center, center + 10)

            if self.point is not None:
                px, py = self.point
                if x0 <= px < x0 + side and y0 <= py < y0 + side:
                    mx = (px - x0) * MAG_SCALE + MAG_SCALE // 2
                    my = (py - y0) * MAG_SCALE + MAG_SCALE // 2
                    r = CIRCLE_RADIUS * MAG_SCALE
                    painter.setPen(QPen(GREEN, 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(mx - r, my - r, r * 2, r * 2)
        finally:
            painter.end()

        self.magnifier.setPixmap(pix)

    # ------------------------------------------------------------ Lifecycle

    def closeEvent(self, event) -> None:
        self._save_current_label()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = LabelerWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
