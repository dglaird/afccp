"""Generate the code reference pages and structured navigation."""

import sys
import os
from pathlib import Path
import mkdocs_gen_files

# Obtain initial working directory
dir_path = os.getcwd() + '/'

# Get main afccp folder path
index = dir_path.find('afccp')
dir_path = dir_path[:index + 6]

# Update working directory
os.chdir(dir_path)

# Ensure 'afccp' is in PYTHONPATH
sys.path.insert(0, os.path.abspath("afccp"))

# Define paths
ROOT_DIR = Path(__file__).parent.parent.parent  # Root of the project
SRC_DIR = ROOT_DIR / "afccp/afccp"  # Path to the source code
OUTPUT_DIR = Path("reference")  # Where to store generated docs

nav = mkdocs_gen_files.Nav()  # Initialize navigation

# Loop through all Python files in the 'afccp' module
for path in sorted(SRC_DIR.rglob("*.py")):
    module_path = path.relative_to(SRC_DIR).with_suffix("")  # e.g., castle/module.py -> module
    doc_path = path.relative_to(SRC_DIR).with_suffix(".md")  # e.g., module.md
    full_doc_path = OUTPUT_DIR / doc_path

    parts = list(module_path.parts)  # Split into module parts

    # Handle __init__.py and __main__.py cases
    if parts[-1] == "__init__":
        parts.pop()  # Remove __init__ from navigation
    elif parts[-1] == "__main__":
        continue  # Skip __main__.py
    elif 'executables' in f'{path}' or 'instances' in f'{path}' or parts[-1] == 'setup':
        continue  # Skip other folders

    # **Fix:** Skip if parts is empty after removing `__init__.py`
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()  # Add to navigation
    module_name = ".".join(parts)

    # Generate API reference markdown
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print(f"::: {module_name}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT_DIR))

# Write structured navigation to SUMMARY.md for mkdocs-literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("\n✅ API reference and structured navigation generated successfully!")