"""Add "Video Compressor" to the Windows right-click > Send to menu (and optionally the Start menu).

    python install_sendto.py               # Send to entry
    python install_sendto.py --start-menu  # also a Start menu entry
    python install_sendto.py --remove      # remove what this script created
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "Video Compressor.pyw")
ICON = os.path.join(HERE, "vidcomp", "assets", "icon.ico")
NAME = "Video Compressor.lnk"


def make_icon():
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPolygon
    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 - needed for painting
    img = QImage(256, 256, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(4, 4)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#6d5cff"))
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setBrush(QColor("white"))
    p.drawPolygon(QPolygon([QPoint(24, 18), QPoint(24, 46), QPoint(46, 32)]))
    p.end()
    img.save(ICON, "ICO")


def pythonw() -> str:
    exe = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return exe if os.path.exists(exe) else sys.executable


def create_shortcut(folder: str):
    path = os.path.join(folder, NAME)
    ps = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK_PATH);"
        "$s.TargetPath = $env:LNK_TARGET; $s.Arguments = $env:LNK_ARGS;"
        "$s.WorkingDirectory = $env:LNK_DIR; $s.IconLocation = $env:LNK_ICON;"
        "$s.Description = 'Compress videos to a target size'; $s.Save()"
    )
    env = dict(os.environ, LNK_PATH=path, LNK_TARGET=pythonw(), LNK_ARGS=f'"{LAUNCHER}"',
               LNK_DIR=HERE, LNK_ICON=ICON)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env)
    print(f"created {path}")


def main():
    if os.name != "nt":
        sys.exit("Windows only")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-menu", action="store_true")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    appdata = os.environ["APPDATA"]
    folders = [os.path.join(appdata, r"Microsoft\Windows\SendTo")]
    if a.start_menu or a.remove:
        folders.append(os.path.join(appdata, r"Microsoft\Windows\Start Menu\Programs"))
    if a.remove:
        for f in folders:
            p = os.path.join(f, NAME)
            if os.path.exists(p):
                os.remove(p)
                print(f"removed {p}")
        return
    if not os.path.exists(ICON):
        make_icon()
    for f in folders:
        create_shortcut(f)


if __name__ == "__main__":
    main()
