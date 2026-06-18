# Configuration file for the HJCD-IK Sphinx documentation builder.
#
# The API reference is produced by Doxygen (XML) and surfaced through Breathe.
# See ../Doxyfile and ../Makefile. Stack/versions mirror the GLASS docs.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "HJCD-IK"
copyright = "2025, A2R Lab"
author = "A2R Lab"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "myst_parser",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 4

# -- Breathe (Doxygen bridge) ------------------------------------------------
breathe_projects = {"hjcdik": "../doxygen/xml"}
breathe_default_project = "hjcdik"
breathe_domain_by_extension = {"cuh": "cpp", "cu": "cpp", "h": "cpp"}

templates_path = ["_templates"]
exclude_patterns = []

# Reopened sub-namespaces across headers can trigger cosmetic duplicate-target
# notices; they are harmless and the rendered HTML is correct.
suppress_warnings = ["docutils"]
numfig = True

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 4,
    "github_url": "https://github.com/A2R-Lab/HJCD-IK",
    "collapse_navigation": True,
}
html_context = {
    "display_github": True,
    "github_user": "A2R-Lab",
    "github_repo": "HJCD-IK",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "doc_path": "docs/source",
}
