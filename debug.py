#!/usr/bin/env python3
"""Entry point for the Triforce Debugger Qt GUI."""

import sys
import argparse

from PySide6.QtWidgets import QApplication

from triforce_debugger.main_window import MainWindow


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Triforce Debugger")
    parser.add_argument("--path", type=str, default=".",
                        help="Directory to scan for .pt model files")
    return parser.parse_args()


def main():
    """Launch the debugger application."""
    args = parse_args()
    _ = args  # Will be used when model browser is wired up

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
