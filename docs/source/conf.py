# Configuration file for the HJCD-IK Sphinx documentation builder.
#
# The API reference is produced by Doxygen (XML) and surfaced through Breathe.
# See ../Doxyfile and ../Makefile. Stack/versions mirror the GLASS docs.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "HJCD-IK"
copyright = "2026, A2R Lab"
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
html_favicon = "_static/favicon/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# The docs are served at <site>/docs/; the project landing page lives at the site root.
html_baseurl = "https://a2r-lab.org/HJCD-IK/docs/"
html_theme_options = {
    "navigation_depth": 4,
    "github_url": "https://github.com/A2R-Lab/HJCD-IK",
    "use_edit_page_button": True,
    "logo": {
        "image_light": "_static/a2r_lab.png",
        "image_dark": "_static/a2r_lab.png",
    },
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
    # Quick link back to the project landing page (the landing lives at the site root).
    "external_links": [
        {"name": "Project page", "url": "https://a2r-lab.org/HJCD-IK/"},
    ],
}
html_context = {
    "display_github": True,
    "github_user": "A2R-Lab",
    "github_repo": "HJCD-IK",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "doc_path": "docs/source",
}
