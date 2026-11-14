"""@TODO."""

from __future__ import annotations

import argparse
import contextlib
import os
import plistlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from fouri import report

SHORTCUT: Final[str] = "4i Pipeline"
SUFFIX_LINUX: Final[str] = ".desktop"
SUFFIX_MACOS: Final[str] = ".app"
SUFFIX_WINDOWS: Final[str] = ".lnk"

LAUNCHER_FILE: Final[Path] = Path("fouri-pipeline")
LAUNCHER_WINDOWS: Final[Path] = Path("Scripts") / LAUNCHER_FILE.with_suffix(".exe")
LAUNCHER_UNIX: Final[Path] = Path("bin") / LAUNCHER_FILE

VENV: Final[str] = ".venv"

ICON_DIR: Final[Path] = Path(__file__).resolve().parent / "icons"
ICON_FILE: Final[Path] = Path("fouri-pipeline")


def shortcut_targets() -> list[Path]:
    """@TODO."""
    match sys.platform:
        case "win32":
            import win32com.client  # pyright: ignore[reportMissingModuleSource]

            shell = win32com.client.Dispatch("WScript.shell")

            desktop_win32 = (Path(shell.SpecialFolders("Desktop")) / SHORTCUT).with_suffix(SUFFIX_WINDOWS)
            programs = (Path(shell.SpecialFolders("Programs")) / SHORTCUT).with_suffix(SUFFIX_WINDOWS)

            return [programs, desktop_win32]

        case "linux":
            menu = (
                (Path(xdg) if (xdg := os.environ.get("XDG_DATA_HOME")) else Path.home() / ".local" / "share")
                / "applications"
                / SHORTCUT
            ).with_suffix(SUFFIX_LINUX)
            try:
                xdg_user_dir = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True)
                desktop_linux = Path(xdg_user_dir.stdout.strip())
            except FileNotFoundError:
                desktop_linux = Path.home() / "Desktop"

            if not desktop_linux.exists():
                return [menu]

            desktop_linux /= SHORTCUT
            desktop_linux = desktop_linux.with_suffix(SUFFIX_LINUX)
            return [menu, desktop_linux]

        case "darwin":
            applications = (Path.home() / "Applications" / SHORTCUT).with_suffix(SUFFIX_MACOS)
            desktop_macos = (Path.home() / "Desktop" / SHORTCUT).with_suffix(SUFFIX_MACOS)

            return [applications, desktop_macos]

        case _:
            return []


def launcher_path(clone: Path) -> Path:
    """@TODO."""
    path = clone / VENV
    return path / LAUNCHER_WINDOWS if sys.platform == "win32" else path / LAUNCHER_UNIX


def icon_file() -> Path:
    match sys.platform:
        case "win32":
            return (ICON_DIR / ICON_FILE).with_suffix(".ico")
        case "darwin":
            return (ICON_DIR / ICON_FILE).with_suffix(".icns")
        case _:
            return (ICON_DIR / ICON_FILE).with_suffix(".png")


def write_lnk(target: Path, launcher: Path, working: Path, icon: Path) -> None:
    """@TODO."""
    import win32com.client  # pyright: ignore[reportMissingModuleSource]

    shell = win32com.client.Dispatch("WScript.shell")
    shortcut = shell.CreateShortcut(str(target))
    shortcut.TargetPath = str(launcher)
    shortcut.WorkingDirectory = str(working)
    shortcut.IconLocation = str(icon)
    shortcut.Save()


def write_desktop_entry(target: Path, launcher: Path, working: Path, icon: Path) -> None:
    """@TODO."""
    entry = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=4i Pipeline",
        f'Exec="{launcher}"',
        f"Path={working}",
        f"Icon={icon}",
        "Terminal=false",
    ]
    entry_text = "\n".join(entry) + "\n"
    target.write_text(entry_text, "utf-8")


def write_app_bundle(bundle: Path, launcher: Path, working: Path, icon: Path) -> None:
    if bundle.exists():
        shutil.rmtree(bundle)
    contents = bundle / "Contents"
    layout = contents / "MacOS"
    layout.mkdir(parents=True, exist_ok=True)
    resources = contents / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon, resources / icon.name)

    info_plist = {
        "CFBundleName": "4i Pipeline",
        "CFBundleIdentifier": "io.github.purvislabteam.fouri-pipeline",
        "CFBundleExecutable": "fouri-pipeline",
        "CFBundlePackageType": "APPL",
        "CFBundleIconFile": icon.name,
    }

    with open(contents / "Info.plist", "wb") as fh:
        plistlib.dump(info_plist, fh)

    script = ["#!/bin/sh", f'cd "{working}" && exec "{launcher}"']
    script_text = "\n".join(script) + "\n"
    script_path = layout / LAUNCHER_FILE
    with open(script_path, "w") as fh:
        fh.write(script_text)
    os.chmod(script_path, mode=0o755)


def make_shortcuts(launcher: Path, working: Path) -> list[Path]:
    shortcuts = shortcut_targets()
    if not shortcuts:
        return shortcuts

    shortcuts[0].parent.mkdir(parents=True, exist_ok=True)
    icon = icon_file()

    match sys.platform:
        case "win32":
            for path in shortcuts:
                write_lnk(path, launcher, working, icon)
        case "linux":
            for p, path in enumerate(shortcuts):
                write_desktop_entry(path, launcher, working, icon)
                if p == 1:
                    os.chmod(path, 0o755)
                    with contextlib.suppress(FileNotFoundError):
                        subprocess.run(
                            ["gio", "set", str(path), "metadata::trusted", "true"], capture_output=True, text=True
                        )
        case "darwin":
            write_app_bundle(shortcuts[0], launcher, working, icon)
            if len(shortcuts) > 1:
                desktop = shortcuts[1]
                if desktop.exists() or desktop.is_symlink():
                    desktop.unlink()
                desktop.symlink_to(shortcuts[0])

        case _:
            pass

    return shortcuts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fouri-shortcuts", description="Make the 4i pipeline desktop and menu shortcuts."
    )
    parser.parse_args(argv)
    if sys.platform not in ("win32", "linux", "darwin"):
        report.say(
            "Shortcuts can only be made on Windows, Linux, and macOS. Start the app with: uv run fouri-pipeline"
        )
        return 1
    clone = Path(__file__).resolve().parents[3]
    launcher = launcher_path(clone)
    if not launcher.exists():
        report.say("The app's launcher is missing: run uv sync in this directory first, then try again.")
        return 1
    created = make_shortcuts(launcher, clone)
    if len(created) > 1:
        report.say(f"Created 4i Pipeline shortcuts: {', '.join(map(str, created))}")
    else:
        report.say(
            f"Made the 4i Pipeline menu shortcut: {created[0]}. No desktop shortcut was created because "
            "the Desktop directory could not be found."
        )

    return 0
