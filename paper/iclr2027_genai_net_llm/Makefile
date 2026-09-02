LATEX=pdflatex
PYTHON=../../.venv/bin/python
MAIN=main
BUILD=build
PROMPT_APPENDIX=generated/contract_v2_prompts_appendix.tex

.PHONY: all audit prompts draft clean

all: audit $(BUILD)/$(MAIN).pdf

audit:
	PYTHONPATH=../.. $(PYTHON) audit_contract_v2_readiness.py

prompts:
	PYTHONPATH=../.. $(PYTHON) ../../comparisons/llm_crn_generation/generate_contract_v2_prompt_review.py

draft:
	@printf '%s\n' 'Draft build blocked: contract-v2 numerical assets do not exist.' >&2
	@printf '%s\n' 'See EXPERIMENT_RECOVERY_PLAN.md and PRE_MASK_QUARANTINE.md.' >&2
	@exit 1

$(BUILD)/$(MAIN).pdf: $(MAIN).tex macros.tex references.bib sections/*.tex figures/*.pdf $(PROMPT_APPENDIX)
	@mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	bibtex $(BUILD)/$(MAIN)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex

clean:
	rm -f $(BUILD)/*.aux $(BUILD)/*.bbl $(BUILD)/*.blg $(BUILD)/*.log
	rm -f $(BUILD)/*.out $(BUILD)/*.pdf $(BUILD)/*.toc
