"""
ui/desktop_app.py
-----------------
PyQt6 desktop frontend for GlobAI V2.
"""

from __future__ import annotations

import html
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any



def _resolve_project_root() -> Path:
    env_root = os.environ.get("GLOBAI_PROJECT_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir, exe_dir.parent, exe_dir.parent.parent, exe_dir.parent.parent.parent])
    candidates.append(Path(__file__).resolve().parent.parent)

    for candidate in candidates:
        if candidate and (candidate / "config.yaml").exists():
            return candidate

    return candidates[0] if candidates else Path.cwd()


PROJECT_ROOT = _resolve_project_root()

sys.path.insert(0, str(PROJECT_ROOT))


from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QCloseEvent, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplashScreen,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from app import (
    APP_NAME,
    CONFIG_PATH,
    DEFAULT_CONFIG,
    MODE_CODER,
    MODE_IMAGE,
    MODE_RAG,
)
from ui.worker import TaskWorker


LOG_PATH = PROJECT_ROOT / "startup_debug.log"
ICON_IMAGE_PATH = PROJECT_ROOT / "ui" / "icon.png"
ICON_ICO_PATH = PROJECT_ROOT / "assets" / "icons" / "GlobAI.ico"
ICON_PATH = ICON_ICO_PATH if ICON_ICO_PATH.exists() else ICON_IMAGE_PATH


def get_windows_username() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return "User"


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #000000;
    color: #ffffff;
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 14px;
}
QFrame#Sidebar, QFrame#SettingsPanel {
    background-color: #030303;
    border: 1px solid #17181d;
}
QFrame#Sidebar {
    border-left: none;
    border-top: none;
    border-bottom: none;
}
QFrame#SettingsPanel {
    border-top: none;
    border-right: none;
    border-bottom: none;
}
QFrame#GlassCard, QFrame#InputShell, QFrame#MessageBubble, QFrame#ImagePanel {
    background-color: rgba(18, 19, 24, 210);
    border: 1px solid #20222a;
    border-radius: 16px;
}
QFrame#MessageBubble {
    border-radius: 18px;
}
QFrame#MessageBubble[user="true"] {
    background-color: rgba(23, 24, 29, 230);
    border: 1px solid #252832;
}
QFrame#MessageBubble[bot="true"] {
    background-color: rgba(12, 13, 17, 235);
    border: 1px solid #1b1e26;
}
QLabel#AppTitle {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#SectionLabel {
    color: #d7d7dc;
    font-size: 14px;
    font-weight: 600;
}
QLabel#MutedLabel {
    color: #8c8f99;
    font-size: 12px;
}
QLabel#MetricValue {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}
QLabel#StatusDot {
    color: #74f088;
    font-size: 20px;
}
QLabel#StatusText {
    color: #dfe2ea;
    font-size: 13px;
    font-weight: 600;
}
QPushButton {
    background-color: rgba(18, 19, 24, 230);
    border: 1px solid #242733;
    border-radius: 13px;
    color: #f5f5f7;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(28, 30, 38, 245);
    border: 1px solid #545a6c;
}
QPushButton:pressed {
    background-color: #0f1015;
}
QPushButton:disabled {
    color: #666a75;
    border: 1px solid #171820;
    background-color: #090a0e;
}
QPushButton#PrimaryButton {
    background-color: #f4f5f8;
    border: 1px solid #ffffff;
    color: #000000;
}
QPushButton#PrimaryButton:hover {
    background-color: #ffffff;
}
QPushButton#ModeButton {
    min-height: 28px;
}
QPushButton#ModeButton[active="true"] {
    background-color: #ffffff;
    border: 1px solid #ffffff;
    color: #000000;
}
QPushButton#GhostButton {
    background-color: transparent;
    border: 1px solid #1d2028;
    color: #b7bbc7;
}
QLineEdit#ChatInput {
    background-color: transparent;
    border: none;
    color: #ffffff;
    font-size: 16px;
    padding: 6px 10px;
}
QLineEdit#ChatInput:disabled {
    color: #6a6e78;
}
QTextEdit#PromptEdit {
    background-color: rgba(14, 15, 19, 240);
    border: 1px solid #20232c;
    border-radius: 16px;
    color: #ffffff;
    padding: 16px;
    selection-background-color: #7d55ff;
}
QTextEdit#ImagePromptEdit {
    background-color: rgba(8, 9, 12, 245);
    border: 1px solid #20232c;
    border-radius: 14px;
    color: #ffffff;
    padding: 12px;
    selection-background-color: #7d55ff;
}
QTextEdit#ImagePromptEdit:disabled {
    color: #686c76;
    border: 1px solid #171820;
    background-color: #06070a;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea QWidget {
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    margin: 8px 2px 8px 2px;
}
QScrollBar::handle:vertical {
    background-color: #282b35;
    border-radius: 5px;
    min-height: 44px;
}
QScrollBar::handle:vertical:hover {
    background-color: #4c5263;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QSlider::groove:horizontal {
    height: 5px;
    background-color: #20222a;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background-color: #ffffff;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #ffffff;
    border: 2px solid #dfe2ff;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QProgressBar {
    background-color: #0b0c10;
    border: 1px solid #222530;
    border-radius: 5px;
    height: 9px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7d55ff, stop:0.52 #ffffff, stop:1 #00eaff);
}
"""


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )


def _global_exception_hook(exctype, value, tb) -> None:
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    logging.critical("UNCAUGHT EXCEPTION: %s", err_msg)
    try:
        if QApplication.instance():
            QMessageBox.critical(None, "Fatal Error", f"An unexpected error occurred:\n{value}")
    finally:
        sys.__excepthook__(exctype, value, tb)


sys.excepthook = _global_exception_hook


def _add_glow(widget: QWidget, color: str = "#7d55ff", blur: int = 24, alpha: int = 90) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setColor(QColor(color + f"{alpha:02x}" if len(color) == 7 else color))
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)


class ModeButton(QPushButton):
    def __init__(self, label: str, mode: str):
        super().__init__(label)
        self.mode = mode
        self.setObjectName("ModeButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", "false")

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class LoadingOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            LoadingOverlay {
                background-color: rgba(0, 0, 0, 176);
            }
            QLabel#OverlayTitle {
                color: #ffffff;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#OverlayDetail {
                color: #aeb3c1;
                font-size: 13px;
            }
            """
        )
        self._dots = 0
        self._base_text = "Working"
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("GlassCard")
        card.setFixedWidth(360)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(12)

        self.title = QLabel("Working")
        self.title.setObjectName("OverlayTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.title)

        self.detail = QLabel("Please keep this window open.")
        self.detail.setObjectName("OverlayDetail")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        card_layout.addWidget(self.detail)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        card_layout.addWidget(self.progress)
        _add_glow(card, "#7d55ff", 36, 80)
        layout.addWidget(card)

        self.timer = QTimer(self)
        self.timer.setInterval(420)
        self.timer.timeout.connect(self._animate)
        self.hide()

    def _animate(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.title.setText(self._base_text + ("." * self._dots))

    def show_state(self, title: str, detail: str = "Please keep this window open.") -> None:
        self._base_text = title
        self._dots = 0
        self.title.setText(title)
        self.detail.setText(detail)
        self.show()
        self.raise_()
        self.timer.start()

    def hide_state(self) -> None:
        self.timer.stop()
        self.hide()


class MessageWidget(QFrame):
    def __init__(self, sender: str, text: str, icon_path: Path | None = None):
        super().__init__()
        self.setObjectName("MessageRow")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 8, 0, 8)
        row.setSpacing(16)

        avatar = QLabel()
        avatar.setFixedSize(52, 52)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            "background-color:#15161c; border:1px solid #272a34; border-radius:26px; color:#ffffff; font-weight:700;"
        )
        if sender == "Bot" and icon_path and icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(
                34,
                34,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            avatar.setPixmap(pix)
        elif sender == "User":
            avatar.setText("U")
        else:
            avatar.setText("G")

        bubble = QFrame()
        bubble.setObjectName("MessageBubble")
        bubble.setProperty("user", "true" if sender == "User" else "false")
        bubble.setProperty("bot", "true" if sender == "Bot" else "false")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(18, 14, 18, 14)
        bubble_layout.setSpacing(8)

        title = QLabel(get_windows_username() if sender == "User" else ("GlobAI" if sender == "Bot" else "System"))
        title.setObjectName("SectionLabel")
        bubble_layout.addWidget(title)

        body = QLabel(self._format_message(text))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setStyleSheet("color:#f1f2f5; font-size:15px; line-height:1.5;")
        bubble_layout.addWidget(body)

        row.addWidget(avatar)
        row.addWidget(bubble, stretch=1)
        row.addStretch(1)

    def _format_message(self, text: str) -> str:
        safe = html.escape(str(text or ""))
        safe = re.sub(
            r"```(?:python)?\n?(.*?)```",
            lambda m: (
                "<pre style='background:#07080b; border:1px solid #262a35; "
                "border-radius:10px; padding:12px; white-space:pre-wrap;'>"
                f"{m.group(1)}</pre>"
            ),
            safe,
            flags=re.DOTALL,
        )
        safe = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe)
        return safe.replace("\n", "<br/>")


class SettingsPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("SettingsPanel")
        self.setFixedWidth(360)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 34, 28, 28)
        layout.setSpacing(18)

        header = QLabel("Settings")
        header.setObjectName("AppTitle")
        layout.addWidget(header)

        prompt_label = QLabel("System Prompt")
        prompt_label.setObjectName("SectionLabel")
        layout.addWidget(prompt_label)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setObjectName("PromptEdit")
        self.prompt_edit.setMinimumHeight(250)
        self.prompt_edit.setPlaceholderText("Define the system-wide instructions for all modes.")
        layout.addWidget(self.prompt_edit)

        self.temp_slider, self.temp_value = self._add_slider(
            layout,
            "Temperature",
            0,
            100,
            0,
            lambda value: f"{value / 100:.2f}",
        )
        self.sim_slider, self.sim_value = self._add_slider(
            layout,
            "Similarity Threshold",
            0,
            100,
            24,
            lambda value: f"{value / 100:.2f}",
        )
        self.topk_slider, self.topk_value = self._add_slider(
            layout,
            "Retrieval Count",
            1,
            6,
            3,
            str,
        )

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.apply_btn = QPushButton("Apply")
        self.save_btn = QPushButton("Save to config")
        self.save_btn.setObjectName("PrimaryButton")
        self.reset_btn = QPushButton("Reset")
        button_row.addWidget(self.apply_btn)
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.reset_btn)
        layout.addLayout(button_row)

        self.status_label = QLabel("Changes can be applied for this session or saved to config.")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _add_slider(self, layout: QVBoxLayout, title: str, low: int, high: int, value: int, fmt):
        label_row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("MutedLabel")
        value_label = QLabel(fmt(value))
        value_label.setObjectName("StatusText")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        label_row.addWidget(label)
        label_row.addWidget(value_label)
        layout.addLayout(label_row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: value_label.setText(fmt(v)))
        layout.addWidget(slider)
        return slider, value_label

    def load_from_backend(self, backend: Any) -> None:
        try:
            prompt = backend.get_system_prompt()
        except Exception:
            prompt = str(backend.config.get("system_prompt", DEFAULT_CONFIG["system_prompt"]))
        self.prompt_edit.setPlainText(prompt)
        self.temp_slider.setValue(int(float(backend.config.get("temperature", 0.0)) * 100))
        self.sim_slider.setValue(int(float(backend.config.get("similarity_threshold", 0.24)) * 100))
        self.topk_slider.setValue(int(backend.config.get("top_k", 3)))

    def values(self) -> dict[str, Any]:
        return {
            "system_prompt": self.prompt_edit.toPlainText().strip(),
            "temperature": self.temp_slider.value() / 100,
            "similarity_threshold": self.sim_slider.value() / 100,
            "top_k": self.topk_slider.value(),
        }


class GlobAIMainWindow(QMainWindow):
    def __init__(self, app_backend):
        super().__init__()
        logging.info("GlobAIMainWindow: __init__ started")
        self.app_backend = app_backend
        self.worker: TaskWorker | None = None
        self.mode_buttons: dict[str, ModeButton] = {}
        self.current_image_path: str | None = None
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 880)
        self.setMinimumSize(1180, 720)
        self.setStyleSheet(STYLESHEET)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.idle_timer = QTimer(self)
        self.idle_timer.setInterval(120000)
        self.idle_timer.timeout.connect(self.on_idle_timeout)
        self.idle_timer.start()

        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(2000)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root = QHBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_workspace(), stretch=1)

        self.settings_panel = SettingsPanel()
        self.settings_panel.load_from_backend(self.app_backend)
        self.settings_panel.apply_btn.clicked.connect(lambda: self.on_apply_settings(False))
        self.settings_panel.save_btn.clicked.connect(lambda: self.on_apply_settings(True))
        self.settings_panel.reset_btn.clicked.connect(self.on_reset_settings)
        root.addWidget(self.settings_panel)

        self.overlay = LoadingOverlay(self.workspace)
        self.add_message("System", f"Welcome back, {get_windows_username()}. GlobAI is ready.")
        self.update_ui_for_mode(self.app_backend.mode)
        self.update_stats()
        logging.info("GlobAIMainWindow: init_ui completed")

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(330)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(30, 32, 28, 28)
        layout.setSpacing(20)

        brand_row = QHBoxLayout()
        logo = QLabel()
        logo.setFixedSize(54, 54)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if ICON_IMAGE_PATH.exists():
            logo.setPixmap(
                QPixmap(str(ICON_IMAGE_PATH)).scaled(
                    48,
                    48,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        brand = QLabel(APP_NAME)
        brand.setObjectName("AppTitle")
        brand_row.addWidget(logo)
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)

        health = QFrame()
        health.setObjectName("GlassCard")
        health_layout = QVBoxLayout(health)
        health_layout.setContentsMargins(18, 16, 18, 16)
        health_layout.setSpacing(10)
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Runtime Status"))
        status_row.addStretch(1)
        dot = QLabel("*")
        dot.setObjectName("StatusDot")
        status_row.addWidget(dot)
        self.runtime_label = QLabel("Ready")
        self.runtime_label.setObjectName("StatusText")
        status_row.addWidget(self.runtime_label)
        health_layout.addLayout(status_row)

        self.kb_value = QLabel("0 vectors")
        self.kb_value.setObjectName("MetricValue")
        health_layout.addWidget(QLabel("Knowledge Base"))
        health_layout.addWidget(self.kb_value)

        self.ram_value = QLabel("RAM --")
        self.ram_value.setObjectName("MutedLabel")
        health_layout.addWidget(self.ram_value)
        _add_glow(health, "#00eaff", 28, 32)
        layout.addWidget(health)

        layout.addWidget(self._section_label("Mode Selector"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        for label, mode in (("RAG", MODE_RAG), ("Coder", MODE_CODER), ("Image", MODE_IMAGE)):
            btn = ModeButton(label, mode)
            btn.clicked.connect(lambda checked=False, m=mode: self.on_mode_button_clicked(m))
            self.mode_buttons[mode] = btn
            mode_row.addWidget(btn)
        layout.addLayout(mode_row)

        layout.addWidget(self._section_label("Model Parameters"))
        self.sidebar_sim_slider, self.sidebar_sim_value = self._sidebar_slider(
            layout,
            "Similarity Threshold",
            0,
            100,
            int(float(self.app_backend.config.get("similarity_threshold", 0.24)) * 100),
            lambda v: f"{v / 100:.2f}",
        )
        self.sidebar_temp_slider, self.sidebar_temp_value = self._sidebar_slider(
            layout,
            "Temperature",
            0,
            100,
            int(float(self.app_backend.config.get("temperature", 0.0)) * 100),
            lambda v: f"{v / 100:.2f}",
        )
        self.sidebar_sim_slider.valueChanged.connect(self.settings_panel_sync_from_sidebar_later)
        self.sidebar_temp_slider.valueChanged.connect(self.settings_panel_sync_from_sidebar_later)

        layout.addStretch(1)

        self.upload_btn = QPushButton("Upload Documents")
        self.upload_btn.setObjectName("GhostButton")
        self.upload_btn.clicked.connect(self.on_upload)
        layout.addWidget(self.upload_btn)

        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.setObjectName("GhostButton")
        self.clear_btn.clicked.connect(self.on_clear_chat)
        layout.addWidget(self.clear_btn)

        return sidebar

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionLabel")
        return label

    def _sidebar_slider(self, layout: QVBoxLayout, title: str, low: int, high: int, value: int, fmt):
        row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("MutedLabel")
        value_label = QLabel(fmt(value))
        value_label.setObjectName("StatusText")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(label)
        row.addWidget(value_label)
        layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: value_label.setText(fmt(v)))
        layout.addWidget(slider)
        return slider, value_label

    def _build_workspace(self) -> QWidget:
        self.workspace = QWidget()
        layout = QVBoxLayout(self.workspace)
        layout.setContentsMargins(30, 28, 30, 28)
        layout.setSpacing(18)

        top_row = QHBoxLayout()
        self.workspace_title = QLabel("Chat")
        self.workspace_title.setObjectName("AppTitle")
        top_row.addWidget(self.workspace_title)
        top_row.addStretch(1)
        self.activity_label = QLabel("Idle")
        self.activity_label.setObjectName("StatusText")
        top_row.addWidget(self.activity_label)
        layout.addLayout(top_row)

        self.activity_progress = QProgressBar()
        self.activity_progress.setRange(0, 0)
        self.activity_progress.hide()
        layout.addWidget(self.activity_progress)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_chat_page())
        self.stack.addWidget(self._build_image_page())
        layout.addWidget(self.stack, stretch=1)

        self.input_shell = QFrame()
        self.input_shell.setObjectName("InputShell")
        input_layout = QHBoxLayout(self.input_shell)
        input_layout.setContentsMargins(16, 12, 12, 12)
        input_layout.setSpacing(10)

        self.input_upload_btn = QPushButton("Upload")
        self.input_upload_btn.setObjectName("GhostButton")
        self.input_upload_btn.clicked.connect(self.on_upload)
        input_layout.addWidget(self.input_upload_btn)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("ChatInput")
        self.input_field.setPlaceholderText("Chat input...")
        self.input_field.returnPressed.connect(self.on_send)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("PrimaryButton")
        self.send_btn.clicked.connect(self.on_send)
        input_layout.addWidget(self.send_btn)
        _add_glow(self.input_shell, "#ffffff", 26, 30)
        layout.addWidget(self.input_shell)
        return self.workspace

    def _build_chat_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(8, 8, 14, 8)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_container)
        page_layout.addWidget(self.scroll_area)
        return page

    def _build_image_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        panel = QFrame()
        panel.setObjectName("ImagePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 22, 22, 22)
        panel_layout.setSpacing(12)

        prompt_header = QLabel("Positive Prompt")
        prompt_header.setObjectName("SectionLabel")
        panel_layout.addWidget(prompt_header)

        self.positive_prompt_edit = QTextEdit()
        self.positive_prompt_edit.setObjectName("ImagePromptEdit")
        self.positive_prompt_edit.setMinimumHeight(92)
        self.positive_prompt_edit.setMaximumHeight(128)
        self.positive_prompt_edit.setPlaceholderText("Describe the image you want to create.")
        panel_layout.addWidget(self.positive_prompt_edit)

        negative_header = QLabel("Negative Prompt")
        negative_header.setObjectName("SectionLabel")
        panel_layout.addWidget(negative_header)

        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setObjectName("ImagePromptEdit")
        self.negative_prompt_edit.setMinimumHeight(72)
        self.negative_prompt_edit.setMaximumHeight(100)
        self.negative_prompt_edit.setPlaceholderText("Describe what to avoid, such as blur, distortion, or unwanted details.")
        panel_layout.addWidget(self.negative_prompt_edit)

        header = QLabel("Image Preview")
        header.setObjectName("SectionLabel")
        panel_layout.addWidget(header)

        self.image_view = QLabel("Generated images appear here.")
        self.image_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_view.setMinimumHeight(300)
        self.image_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_view.setStyleSheet(
            "background-color:#050608; border:1px solid #1b1e26; border-radius:16px; color:#8c8f99;"
        )
        panel_layout.addWidget(self.image_view, stretch=1)

        self.image_path_label = QLabel("No image generated in this session.")
        self.image_path_label.setObjectName("MutedLabel")
        self.image_path_label.setWordWrap(True)
        panel_layout.addWidget(self.image_path_label)
        _add_glow(panel, "#7d55ff", 34, 35)
        layout.addWidget(panel)
        return page

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "overlay"):
            self.overlay.setGeometry(self.workspace.rect())
        if self.current_image_path:
            self._show_image(self.current_image_path)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.on_upload_files(files)

    def settings_panel_sync_from_sidebar_later(self) -> None:
        self.settings_panel.sim_slider.blockSignals(True)
        self.settings_panel.temp_slider.blockSignals(True)
        self.settings_panel.sim_slider.setValue(self.sidebar_sim_slider.value())
        self.settings_panel.temp_slider.setValue(self.sidebar_temp_slider.value())
        self.settings_panel.sim_slider.blockSignals(False)
        self.settings_panel.temp_slider.blockSignals(False)

    def sync_sidebar_from_settings(self) -> None:
        self.sidebar_sim_slider.blockSignals(True)
        self.sidebar_temp_slider.blockSignals(True)
        self.sidebar_sim_slider.setValue(self.settings_panel.sim_slider.value())
        self.sidebar_temp_slider.setValue(self.settings_panel.temp_slider.value())
        self.sidebar_sim_slider.blockSignals(False)
        self.sidebar_temp_slider.blockSignals(False)

    def add_message(self, sender: str, text: str) -> None:
        insert_at = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(insert_at, MessageWidget(sender, text, ICON_IMAGE_PATH))
        QTimer.singleShot(
            0,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

    def on_clear_chat(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.add_message("System", "Conversation cleared.")

    def update_stats(self) -> None:
        try:
            import psutil

            mem = psutil.virtual_memory()
            self.ram_value.setText(f"RAM {mem.percent:.1f}%")
        except Exception:
            self.ram_value.setText("RAM unavailable")
        try:
            count = self.app_backend.rag_system.vector_store.count()
            self.kb_value.setText(f"{count:,} vectors")
        except Exception:
            self.kb_value.setText("Vectors unavailable")

    def update_ui_for_mode(self, mode: str) -> None:
        for key, button in self.mode_buttons.items():
            button.set_active(key == mode)
        self.workspace_title.setText("Image" if mode == MODE_IMAGE else "Chat")
        self.stack.setCurrentIndex(1 if mode == MODE_IMAGE else 0)
        self.input_upload_btn.setVisible(mode != MODE_IMAGE)
        self.upload_btn.setVisible(mode != MODE_IMAGE)
        if mode == MODE_IMAGE:
            self.input_field.setVisible(False)
            self.send_btn.setText("Generate")
        elif mode == MODE_CODER:
            self.input_field.setVisible(True)
            self.send_btn.setText("Send")
            self.input_field.setPlaceholderText("Ask for code, debugging, or implementation help...")
        else:
            self.input_field.setVisible(True)
            self.send_btn.setText("Send")
            self.input_field.setPlaceholderText("Ask a question from your knowledge base...")

    def set_loading(self, is_loading: bool, title: str = "Working", detail: str = "") -> None:
        controls = [
            self.input_field,
            self.send_btn,
            self.upload_btn,
            self.input_upload_btn,
            self.clear_btn,
            self.settings_panel.apply_btn,
            self.settings_panel.save_btn,
            self.settings_panel.reset_btn,
        ]
        if hasattr(self, "positive_prompt_edit"):
            controls.extend([self.positive_prompt_edit, self.negative_prompt_edit])
        controls.extend(self.mode_buttons.values())

        for widget in controls:
            widget.setEnabled(not is_loading)

        if is_loading:
            self.runtime_label.setText("Busy")
            self.activity_label.setText(title)
            self.activity_progress.show()
            self.overlay.setGeometry(self.workspace.rect())
            self.overlay.show_state(title, detail or "The current operation is still running.")
        else:
            self.runtime_label.setText("Ready")
            self.activity_label.setText("Idle")
            self.activity_progress.hide()
            self.overlay.hide_state()

    def on_mode_button_clicked(self, mode: str) -> None:
        if self.app_backend.mode == mode:
            self.update_ui_for_mode(mode)
            return
        self.set_loading(True, f"Loading {mode}", "Switching models and cleaning up previous runtime state.")
        self.worker = TaskWorker(self.app_backend.switch_mode, mode)
        self.worker.finished.connect(self.on_mode_switched)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    @pyqtSlot(object)
    def on_mode_switched(self, result: dict[str, Any]) -> None:
        self.set_loading(False)
        if result.get("ok"):
            self.update_ui_for_mode(self.app_backend.mode)
            self.add_message("System", f"Mode switched to **{self.app_backend.mode}**.")
        else:
            self.add_message("System", f"Mode switch failed: {result.get('error')}")
            self.update_ui_for_mode(self.app_backend.mode)

    def on_send(self) -> None:
        mode = self.app_backend.mode
        negative_prompt = ""
        if mode == MODE_IMAGE:
            prompt = self.positive_prompt_edit.toPlainText().strip()
            negative_prompt = self.negative_prompt_edit.toPlainText().strip()
        else:
            prompt = self.input_field.text().strip()
        if not prompt:
            return

        if mode == MODE_IMAGE:
            display_prompt = prompt
            if negative_prompt:
                display_prompt = f"{prompt}\n\nNegative: {negative_prompt}"
            self.add_message("User", display_prompt)
        else:
            self.input_field.clear()
            self.add_message("User", prompt)

        if mode == MODE_IMAGE:
            title = "Generating image"
            detail = "Stable Diffusion is rendering. This can take a while on CPU."
        elif mode == MODE_CODER:
            title = "Thinking in Coder"
            detail = "The coder model is preparing a response."
        else:
            title = "Thinking in RAG"
            detail = "Retrieving context and preparing an answer."

        self.set_loading(True, title, detail)
        image_kwargs = {"output_dir": PROJECT_ROOT / "data" / "outputs" / "images"}
        if mode == MODE_IMAGE:
            image_kwargs["negative_prompt"] = negative_prompt
        self.worker = TaskWorker(self.app_backend.route, prompt, image_kwargs=image_kwargs)
        self.worker.finished.connect(self.on_generate_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    @pyqtSlot(object)
    def on_generate_finished(self, result: dict[str, Any]) -> None:
        self.set_loading(False)
        if not result.get("ok"):
            self.add_message("System", f"Error: {result.get('error')}")
            return

        if self.app_backend.mode == MODE_IMAGE or result.get("mode") == MODE_IMAGE:
            path = result.get("path")
            if path and os.path.exists(path):
                self._show_image(str(path))
                self.add_message("System", "Image generated successfully.")
            else:
                self.add_message("System", "Image generation completed without a valid image path.")
        else:
            self.add_message("Bot", result.get("answer", "No response."))

    def _show_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_path_label.setText("Generated image could not be loaded.")
            return
        self.current_image_path = path
        target = self.image_view.size()
        scaled = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_view.setPixmap(scaled)
        self.image_path_label.setText(path)

    @pyqtSlot(str)
    def on_error(self, err_msg: str) -> None:
        self.set_loading(False)
        self.add_message("System", f"Core exception: {err_msg}")

    def on_upload(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload Documents",
            "",
            "Documents (*.pdf *.docx *.txt *.pptx);;All files (*.*)",
        )
        if files:
            self.on_upload_files(files)

    def on_upload_files(self, files: list[str]) -> None:
        self.set_loading(True, "Indexing documents", f"Indexing {len(files)} file(s).")
        self.worker = TaskWorker(self.app_backend.index_documents, files)
        self.worker.finished.connect(self.on_index_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    @pyqtSlot(object)
    def on_index_finished(self, result: dict[str, Any]) -> None:
        self.set_loading(False)
        if result.get("ok"):
            self.add_message("System", f"Indexed {result.get('count', 0)} chunk(s).")
            self.update_stats()
        else:
            self.add_message("System", f"Indexing failed: {result.get('error')}")

    def on_apply_settings(self, save: bool) -> None:
        values = self.settings_panel.values()
        prompt = values["system_prompt"] or DEFAULT_CONFIG["system_prompt"]

        try:
            if save:
                self.app_backend.save_system_prompt(prompt)
            else:
                self.app_backend.set_system_prompt(prompt)

            self.app_backend.config["temperature"] = values["temperature"]
            self.app_backend.config["similarity_threshold"] = values["similarity_threshold"]
            self.app_backend.config["top_k"] = values["top_k"]

            try:
                self.app_backend.query_engine.temperature = values["temperature"]
                self.app_backend.hybrid_retriever.threshold = values["similarity_threshold"]
                self.app_backend.hybrid_retriever.top_k = values["top_k"]
            except Exception:
                logging.debug("Runtime setting sync partially skipped.", exc_info=True)

            self.sync_sidebar_from_settings()

            if save:
                self._persist_numeric_settings(values)
                self.settings_panel.status_label.setText("Settings saved to config.yaml.")
                self.add_message("System", "Settings saved to config.")
            else:
                self.settings_panel.status_label.setText("Settings applied for this session.")
                self.add_message("System", "Settings applied for this session.")
        except Exception as exc:
            self.settings_panel.status_label.setText(f"Settings update failed: {exc}")
            self.add_message("System", f"Settings update failed: {exc}")

    def _persist_numeric_settings(self, values: dict[str, Any]) -> None:
        content = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
        replacements = {
            "temperature": f"{values['temperature']:.2f}",
            "similarity_threshold": f"{values['similarity_threshold']:.2f}",
            "top_k": str(int(values["top_k"])),
        }
        for key, value in replacements.items():
            line = f"{key}: {value}"
            if re.search(rf"^{re.escape(key)}\s*:", content, flags=re.MULTILINE):
                content = re.sub(
                    rf"^{re.escape(key)}\s*:.*$",
                    line,
                    content,
                    flags=re.MULTILINE,
                )
            else:
                content = content.rstrip() + f"\n{line}\n"
        CONFIG_PATH.write_text(content, encoding="utf-8")

    def on_reset_settings(self) -> None:
        self.settings_panel.prompt_edit.setPlainText(DEFAULT_CONFIG["system_prompt"])
        self.settings_panel.temp_slider.setValue(int(DEFAULT_CONFIG["temperature"] * 100))
        self.settings_panel.sim_slider.setValue(int(DEFAULT_CONFIG["similarity_threshold"] * 100))
        self.settings_panel.topk_slider.setValue(int(DEFAULT_CONFIG["top_k"]))
        self.sync_sidebar_from_settings()
        self.settings_panel.status_label.setText("Default settings loaded in the panel. Apply or save when ready.")

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.idle_timer.stop()
            self.stats_timer.stop()
            self.set_loading(True, "Closing", "Cleaning up loaded runtime state.")
            QApplication.processEvents()
            self.app_backend._unload_every_runtime("App Closed")
        except Exception:
            logging.debug("Cleanup during close failed.", exc_info=True)
        event.accept()

    def on_idle_timeout(self) -> None:
        if self.app_backend and getattr(self.app_backend, "mode", None) != MODE_IMAGE:
            try:
                from core.memory_manager import MemoryManager

                MemoryManager.hard_cleanup("idle cleanup")
            except Exception:
                logging.debug("Idle cleanup skipped.", exc_info=True)


def _build_splash_pixmap() -> QPixmap:
    if ICON_IMAGE_PATH.exists():
        return QPixmap(str(ICON_IMAGE_PATH)).scaled(
            420,
            420,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    pixmap = QPixmap(420, 240)
    pixmap.fill(QColor("#000000"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, APP_NAME)
    painter.end()
    return pixmap


def main() -> None:
    _configure_logging()
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "GlobAI.Production.Desktop"
            )
        except Exception:
            logging.debug("Unable to set AppUserModelID.", exc_info=True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("GlobAI")
    app.setDesktopFileName("GlobAI.Production.Desktop")
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    splash = QSplashScreen(_build_splash_pixmap(), Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage(
        "Initializing GlobAI...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        QColor("#ffffff"),
    )
    app.processEvents()

    try:
        from app import build_app

        backend = build_app(skip_preflight=getattr(sys, "frozen", False))
        window = GlobAIMainWindow(backend)
        window.show()
        splash.finish(window)
        sys.exit(app.exec())
    except Exception as exc:
        logging.critical("FATAL STARTUP ERROR: %s\n%s", exc, traceback.format_exc())
        splash.finish(None)
        QMessageBox.critical(None, "Startup Error", f"Failed to start GlobAI:\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
