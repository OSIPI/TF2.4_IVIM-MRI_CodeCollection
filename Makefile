# Makefile — one-command reproduction of the IVIM uncertainty/calibration study.
#
# Convention: run from the repo root. The uq/ package reaches into the
# (unmodified) upstream src/ tree via uq/__init__.py, so no PYTHONPATH juggling
# is needed. Use your project virtualenv, e.g.:
#
#     python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
#     make smoke
#
# or point PYTHON at an interpreter directly:
#
#     make smoke PYTHON=.venv/bin/python
#
PYTHON ?= python

.PHONY: all smoke test grid calib figures clean help

help:           ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

smoke:          ## fast green check: tiny calibration cell (bootstrap, no DL) + tests
	W3_SMOKE=1 W3_GROUPS=bootstrap $(PYTHON) -m uq.run_w3_calib
	$(PYTHON) -m pytest uq

test:           ## run the analysis-layer test suite (isolated from upstream tests)
	$(PYTHON) -m pytest uq

grid:           ## full accuracy grid -> ivim_raw_v3.npz / ivim_summary_v3.csv (needs DL stack)
	$(PYTHON) -m uq.run_grid_v3

calib:          ## full calibration campaign -> calib_w3.csv (needs DL stack; long)
	$(PYTHON) -m uq.run_w3_calib

figures:        ## regenerate headline figures from calib_w3.csv into figures/
	$(PYTHON) -m uq.make_figures calib_w3.csv figures

all: grid calib figures   ## full reproduction: grid -> calibration -> figures

clean:          ## remove gitignored run artifacts (keeps committed figures/)
	rm -f calib_w3.csv ivim_raw_v*.npz ivim_summary_v*.csv calib_w1.csv \
	      fig_reliability.png fig_calibration_heatmap.png \
	      reliability_diagrams.jsx calibration_heatmap.jsx
