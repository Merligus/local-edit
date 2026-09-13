"""Checks that the windows can actually talk to each other.

Run:  python3 tests/test_ui_wiring.py

This file exists because of a bug that reached the user. `MainWindow._start`
calls `self._setup.verdict()`; a slice-based edit to a neighbouring method
deleted `verdict()` and `hardware()` from `SetupPage` without touching the call
sites. Nothing noticed. Every other test in this suite called the page's methods
*directly* — none of them ever pressed the button — so the path from a click to
`_start` was never executed, and a Qt slot swallows the AttributeError it raised.
The result was an application whose primary button silently did nothing.

Two kinds of check, because the bug had two chances to be caught and neither
existed:

* a **static** one, by AST, that every `self._page.method()` call in
  `main_window.py` resolves to a method the page defines. It needs no Qt and no
  display and runs in milliseconds;
* a **live** one that constructs the real window and *clicks*, because the
  static check cannot see a signal connected to the wrong slot.
"""

import ast
from pathlib import Path

from _harness import check, run                                    # noqa: E402

UI = Path(__file__).resolve().parents[1] / "local_edit" / "ui"

#: `MainWindow` attribute -> the module and class it holds.
OWNED = {
    "_setup": ("setup_page.py", "SetupPage"),
    "_progress": ("progress_page.py", "ProgressPage"),
    "_result": ("result_page.py", "ResultPage"),
}


def methods_of(filename, classname):
    tree = ast.parse((UI / filename).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == classname:
            names = {n.name for n in node.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            # Signals are class attributes, and are "called" via .connect/.emit.
            names |= {t.id for n in node.body if isinstance(n, ast.Assign)
                      for t in n.targets if isinstance(t, ast.Name)}
            return names
    return set()


def calls_on(attr):
    """Every `self.<attr>.<name>` referenced anywhere in main_window.py."""
    tree = ast.parse((UI / "main_window.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == attr):
            found.add(node.attr)
    return found


def test_main_window_only_calls_what_exists():
    print("\nevery page method the window uses is a method the page has")
    for attr, (filename, classname) in OWNED.items():
        have = methods_of(filename, classname)
        check(bool(have), f"{classname} was parsed from {filename}")
        for name in sorted(calls_on(attr)):
            check(name in have,
                  f"self.{attr}.{name} -> {classname}.{name} exists")


def test_pages_expose_their_signals():
    print("\nthe signals the window connects to are declared")
    for classname, filename, signals in (
            ("SetupPage", "setup_page.py", {"generate_requested"}),
            ("ProgressPage", "progress_page.py", {"cancel_requested"}),
            ("ResultPage", "result_page.py", {"back_requested",
                                              "continue_requested"})):
        have = methods_of(filename, classname)
        for s in signals:
            check(s in have, f"{classname}.{s} is declared")


def test_clicking_generate_starts_a_run():
    print("\nclicking Generate actually starts a run")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        check(True, "PySide6 is not installed; skipping the live check")
        return

    app = QApplication.instance() or QApplication([])
    from local_edit.ui import icons, metrics
    icons.install(app)
    metrics.install(app)
    from local_edit.ui.main_window import MainWindow

    dialogs = []
    QMessageBox.warning = lambda *a, **k: dialogs.append(a[1])

    w = MainWindow()
    w.show()
    s = w._setup
    s.set_prompt("a red bicycle against a white wall")
    app.processEvents()

    check(s._go.isEnabled(), "Generate is enabled once there is a prompt")
    s._go.click()
    app.processEvents()

    if dialogs:
        # No engine on this machine: the window said so, which is the correct
        # behaviour and is itself the thing being checked.
        check(True, f"no engine installed, and the window explained: {dialogs[0]!r}")
    else:
        check(w._stack.currentIndex() == 1,
              "the click moved to the progress page — this is the assertion "
              "that would have caught the missing verdict()")
        check(w._thread is not None, "and started a worker thread")
        w._cancel()
        w._teardown()
        check(w._thread is None, "cancelling tears the thread down")
    w._engine.stop()


if __name__ == "__main__":
    raise SystemExit(run(test_main_window_only_calls_what_exists,
                         test_pages_expose_their_signals,
                         test_clicking_generate_starts_a_run))
