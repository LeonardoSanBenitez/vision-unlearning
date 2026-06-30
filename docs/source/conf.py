import os
import sys
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../../vision_unlearning"))

project = "vision-unlearning"
author = "Leonardo San Benitez Pereira"

try:
    release = _pkg_version("vision-unlearning")
except Exception:
    release = "unknown"

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autoapi_dirs = ["../../vision_unlearning"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"

# Exclude benchmark pipeline scripts from the public API docs.
# I_care and u_care are research pipeline scripts, not public library surface.
autoapi_ignore = [
    "*/benchmarks/I_care/*",
    "*/benchmarks/u_care/*",
    "*/__pycache__/*",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
