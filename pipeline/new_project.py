#!/usr/bin/env python3
"""
new_project.py — small GUI for creating a new experiment project.

Double-click "New Project.desktop" (Linux) or "New Project.command" (macOS),
or run directly:

    python3 new_project.py

Wraps the same scaffolding used by `tracker new-project`, adding a folder
picker, tag handling, and the option to open the project folder when done.
Uses tkinter only (Python standard library), so it needs no extra install.
"""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import __version__
from pipeline._scaffold import (
    print_next_steps,
    resolve_tracker_tag,
    scaffold_project,
)

# Colours roughly matching the tracker app, so the two feel related.
BG        = "#1a1a2e"
PANEL     = "#16213e"
BORDER    = "#0f3460"
TEXT      = "#e0e0e0"
MUTED     = "#8899bb"
ACCENT    = "#a0c4ff"
OK_GREEN  = "#2d6a4f"
OK_TEXT   = "#d8f3dc"
ERR_RED   = "#ff6b6b"
WARN      = "#ffd700"


class NewProjectApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("New tracking project")
        root.configure(bg=BG)
        root.geometry("620x560")
        root.minsize(560, 520)

        self.tracker_tag: str | None = None
        self.tag_error: str | None = None
        self.created_project_dir: Path | None = None

        self._build_ui()
        self._check_tag()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=BG, padx=24, pady=20)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer, text="NEW TRACKING PROJECT", bg=BG, fg=ACCENT,
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            outer,
            text="Creates a shareable project repo that pins this tracker version.",
            bg=BG, fg=MUTED, font=("TkDefaultFont", 9),
        ).pack(anchor="w", pady=(2, 14))

        # --- tracker version panel ---
        self.tag_panel = tk.Frame(outer, bg=PANEL, padx=12, pady=10,
                                  highlightbackground=BORDER, highlightthickness=1)
        self.tag_panel.pack(fill="x", pady=(0, 14))
        self.tag_label = tk.Label(
            self.tag_panel, text="Checking tracker version…",
            bg=PANEL, fg=MUTED, justify="left", anchor="w",
            font=("TkDefaultFont", 9), wraplength=540,
        )
        self.tag_label.pack(fill="x")

        self.tag_fix_frame = tk.Frame(self.tag_panel, bg=PANEL)
        tk.Label(self.tag_fix_frame, text="Create and push tag:",
                 bg=PANEL, fg=MUTED, font=("TkDefaultFont", 9)).pack(side="left")
        self.tag_entry = tk.Entry(self.tag_fix_frame, width=12, bg=BG, fg=TEXT,
                                  insertbackground=TEXT, relief="flat")
        self.tag_entry.pack(side="left", padx=6)
        self.tag_button = tk.Button(
            self.tag_fix_frame, text="Tag and push", command=self._create_tag,
            bg=BORDER, fg=TEXT, relief="flat", padx=10, cursor="hand2",
        )
        self.tag_button.pack(side="left")

        # --- form ---
        form = tk.Frame(outer, bg=BG)
        form.pack(fill="x")

        tk.Label(form, text="Project name", bg=BG, fg=MUTED,
                 font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        name_entry = tk.Entry(form, textvariable=self.name_var, bg=PANEL, fg=TEXT,
                              insertbackground=TEXT, relief="flat")
        name_entry.grid(row=1, column=0, columnspan=2, sticky="ew", ipady=5, pady=(2, 2))
        tk.Label(form, text="Letters, numbers and underscores — becomes the repo name.",
                 bg=BG, fg="#5a7a99", font=("TkDefaultFont", 8)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 12))

        tk.Label(form, text="Location", bg=BG, fg=MUTED,
                 font=("TkDefaultFont", 9)).grid(row=3, column=0, sticky="w")
        self.path_var = tk.StringVar(value=str(Path.home() / "projects"))
        path_entry = tk.Entry(form, textvariable=self.path_var, bg=PANEL, fg=TEXT,
                              insertbackground=TEXT, relief="flat")
        path_entry.grid(row=4, column=0, sticky="ew", ipady=5, pady=(2, 2))
        tk.Button(form, text="Browse…", command=self._browse,
                  bg=BORDER, fg=TEXT, relief="flat", padx=12, cursor="hand2").grid(
            row=4, column=1, padx=(8, 0))
        tk.Label(form, text="The project folder is created inside this location.",
                 bg=BG, fg="#5a7a99", font=("TkDefaultFont", 8)).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 12))

        form.columnconfigure(0, weight=1)

        self.open_after_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            outer, text="Open the project folder when done",
            variable=self.open_after_var, bg=BG, fg=TEXT, selectcolor=PANEL,
            activebackground=BG, activeforeground=TEXT, relief="flat",
            font=("TkDefaultFont", 9), cursor="hand2",
        ).pack(anchor="w", pady=(0, 14))

        self.create_button = tk.Button(
            outer, text="Create project", command=self._create,
            bg=OK_GREEN, fg=OK_TEXT, relief="flat", pady=9, cursor="hand2",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.create_button.pack(fill="x")

        # --- result panel (hidden until something happens) ---
        self.result_frame = tk.Frame(outer, bg=BG)
        self.result_frame.pack(fill="both", expand=True, pady=(14, 0))
        self.result_banner = tk.Label(
            self.result_frame, text="", bg=BG, fg=TEXT, anchor="w",
            font=("TkDefaultFont", 11, "bold"), wraplength=540, justify="left",
        )
        self.result_detail = tk.Label(
            self.result_frame, text="", bg=BG, fg=MUTED, anchor="w",
            font=("TkDefaultFont", 9), wraplength=540, justify="left",
        )

    # ------------------------------------------------------------ actions

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(
            title="Choose where to create the project",
            initialdir=self.path_var.get() or str(Path.home()),
        )
        if chosen:
            self.path_var.set(chosen)

    def _check_tag(self) -> None:
        """Resolve the tracker's git tag and reflect the result in the UI."""
        try:
            self.tracker_tag = resolve_tracker_tag(REPO_ROOT, __version__)
            self.tag_error = None
            self.tag_label.config(
                text=f"Tracker version {self.tracker_tag} — new projects will pin to this.",
                fg=OK_TEXT,
            )
            self.tag_fix_frame.pack_forget()
            self.create_button.config(state="normal", bg=OK_GREEN)
        except RuntimeError as error:
            self.tracker_tag = None
            self.tag_error = str(error)
            self.tag_label.config(text=str(error), fg=WARN)
            self.tag_entry.delete(0, "end")
            self.tag_entry.insert(0, f"v{__version__}")
            self.tag_fix_frame.pack(fill="x", pady=(10, 0))
            self.create_button.config(state="disabled", bg="#3a3a4e")

    def _create_tag(self) -> None:
        """Create and push the tag the user typed, then re-check."""
        tag = self.tag_entry.get().strip()
        if not tag:
            return
        self.tag_button.config(state="disabled", text="Tagging…")

        def work() -> None:
            try:
                subprocess.run(["git", "-C", str(REPO_ROOT), "tag", tag],
                               check=True, capture_output=True, text=True)
                subprocess.run(["git", "-C", str(REPO_ROOT), "push", "origin", tag],
                               check=True, capture_output=True, text=True)
                self.root.after(0, self._check_tag)
            except subprocess.CalledProcessError as error:
                message = (error.stderr or error.stdout or str(error)).strip()
                self.root.after(0, lambda: self.tag_label.config(
                    text=f"Could not create tag:\n{message}", fg=ERR_RED))
            finally:
                self.root.after(0, lambda: self.tag_button.config(
                    state="normal", text="Tag and push"))

        threading.Thread(target=work, daemon=True).start()

    def _create(self) -> None:
        name = self.name_var.get().strip()
        parent = self.path_var.get().strip()

        if not name:
            self._show_result(False, "Enter a project name.")
            return
        if not all(ch.isalnum() or ch == "_" for ch in name):
            self._show_result(
                False, "Invalid project name.",
                "Use only letters, numbers and underscores — it becomes a repo name.",
            )
            return
        if not parent:
            self._show_result(False, "Choose a location.")
            return
        if self.tracker_tag is None:
            self._show_result(False, "Tracker version is not tagged yet.",
                              "Create the tag above first.")
            return

        projects_dir = Path(parent).expanduser().resolve()
        try:
            projects_dir.mkdir(parents=True, exist_ok=True)
            project_dir = scaffold_project(
                project_name=name,
                projects_dir=projects_dir,
                tracker_tag=self.tracker_tag,
                project_template_file=REPO_ROOT / "project_template" / "project.yaml",
            )
        except (RuntimeError, OSError) as error:
            self._show_result(False, "Could not create the project.", str(error))
            return

        self.created_project_dir = project_dir
        print_next_steps(project_dir, name, self.tracker_tag)

        self._show_result(
            True,
            f"Project created — pinned to tracker {self.tracker_tag}",
            f"{project_dir}\n\n"
            f"Next:\n"
            f"  1. cd {project_dir}\n"
            f"  2. git init && git add -A && git commit -m 'Initial scaffold'\n"
            f"  3. gh repo create StefanMPopp/{name} --private --source=. --push\n"
            f"  4. uv sync --extra tracking\n"
            f"  5. uv run jupyter lab 1_pipeline.ipynb",
        )
        self.create_button.config(text="Create another", bg=BORDER, fg=TEXT)

        if self.open_after_var.get():
            self._open_folder(project_dir)

    def _open_folder(self, folder: Path) -> None:
        opener = {"darwin": "open", "win32": "explorer"}.get(sys.platform, "xdg-open")
        try:
            subprocess.Popen([opener, str(folder)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass   # no desktop file manager available; the path is shown anyway

    def _show_result(self, success: bool, banner: str, detail: str = "") -> None:
        self.result_banner.config(
            text=("✓  " if success else "✗  ") + banner,
            fg=(OK_TEXT if success else ERR_RED),
        )
        self.result_detail.config(text=detail)
        self.result_banner.pack(fill="x", anchor="w")
        if detail:
            self.result_detail.pack(fill="x", anchor="w", pady=(6, 0))
        else:
            self.result_detail.pack_forget()


def main() -> None:
    root = tk.Tk()
    NewProjectApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
