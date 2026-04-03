import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../vision_unlearning"))

project = "vision-unlearning"
author = "Leonardo Benitez Pereira, Carolina Kelsch, Natnael Mola, Juan Carlos San Miguel, Luis Herranz"
release = "0.1.8"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
]

autoapi_dirs = ["../../vision_unlearning"]  # Extracts docstrings
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_heading_anchors = 3