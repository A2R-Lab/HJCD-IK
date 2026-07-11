# HJCD-IK documentation build (Sphinx + Doxygen/Breathe).
#
#   make all     # doxygen XML -> sphinx HTML (the full build, used by CI)
#   make html    # sphinx HTML only (assumes doxygen XML already present)
#   make clean

SPHINXBUILD ?= sphinx-build
DOXYGEN     ?= doxygen
SOURCEDIR    = source
BUILDDIR     = build

.PHONY: all doxygen clean help

all: clean doxygen html

doxygen:
	@$(DOXYGEN) Doxyfile

clean:
	@rm -rf $(BUILDDIR)/* doxygen/xml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

# Catch-all: route any other target (e.g. `html`) to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"
