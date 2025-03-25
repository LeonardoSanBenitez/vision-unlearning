import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../vision_unlearning"))

project = "vision-unlearning"
author = ""
release = "0.1.0"

extensions = [
    "myst_parser",  # Enables Markdown support
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

autoapi_dirs = ["../../vision_unlearning"]  # Extracts docstrings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
