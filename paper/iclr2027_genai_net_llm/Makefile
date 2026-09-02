LATEX=pdflatex
PYTHON=../../.venv/bin/python
MAIN=main
BUILD=build

.PHONY: all audit draft clean

all: audit $(BUILD)/$(MAIN).pdf

audit:
	PYTHONPATH=../.. $(PYTHON) audit_contract_v2_readiness.py

draft:
	@printf '%s\n' 'Draft build blocked: contract-v2 numerical assets do not exist.' >&2
	@printf '%s\n' 'See EXPERIMENT_RECOVERY_PLAN.md and PRE_MASK_QUARANTINE.md.' >&2
	@exit 1

$(BUILD)/$(MAIN).pdf: $(MAIN).tex macros.tex references.bib sections/*.tex figures/*.pdf
	@mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	bibtex $(BUILD)/$(MAIN)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex

clean:
	rm -f $(BUILD)/*.aux $(BUILD)/*.bbl $(BUILD)/*.blg $(BUILD)/*.log
	rm -f $(BUILD)/*.out $(BUILD)/*.pdf $(BUILD)/*.toc
