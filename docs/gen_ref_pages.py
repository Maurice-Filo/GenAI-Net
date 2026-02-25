from __future__ import annotations

from pathlib import Path
import mkdocs_gen_files

PKG_NAME = "RL4CRN"

root = Path(__file__).resolve().parent.parent
pkg_dir = root / PKG_NAME

nav = mkdocs_gen_files.Nav()

def dotted_path_for(py_file: Path) -> str:
    """
    Convert a .py file path to a dotted path for mkdocstrings.

    - For __init__.py, return the package path without '.__init__'
      e.g. RL4CRN/agents/__init__.py -> RL4CRN.agents
    - For normal modules, return full module path
      e.g. RL4CRN/utils/ffnn.py -> RL4CRN.utils.ffnn
    """
    rel = py_file.relative_to(root).with_suffix("")  # RL4CRN/utils/ffnn
    parts = list(rel.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]  # drop __init__

    return ".".join(parts)

for py_file in sorted(pkg_dir.rglob("*.py")):
    if "__pycache__" in py_file.parts:
        continue

    dotted = dotted_path_for(py_file)

    rel_under_pkg = py_file.relative_to(pkg_dir)  # e.g. agents/abstract_agent.py

    if py_file.name == "__init__.py":
        doc_path = Path("reference") / rel_under_pkg.parent / "index.md"
        nav_key = [PKG_NAME, *rel_under_pkg.parent.parts]
        title = rel_under_pkg.parent.name if rel_under_pkg.parent.parts else PKG_NAME
    else:
        doc_path = Path("reference") / rel_under_pkg.with_suffix(".md")
        nav_key = [PKG_NAME, *rel_under_pkg.parts]
        title = rel_under_pkg.stem

    nav[nav_key] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"::: {dotted}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: true\n")
        f.write("      members_order: source\n")
        f.write("      inherited_members: true\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
