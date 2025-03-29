import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../vision_unlearning"))

project = "vision-unlearning"
author = ""
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    #"myst_parser",  # Enables Markdown support
]

autoapi_dirs = ["../../vision_unlearning"]  # Extracts docstrings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Add MyST configuration
# myst_enable_extensions = [
#     "dollarmath",
#     "amsmath",
# ]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }
# myst_heading_anchors = 3