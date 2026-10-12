LATEX=pdflatex
PYTHON=../../.venv/bin/python
MAIN=main
BUILD=build
PROMPT_APPENDIX=generated/contract_v2_prompts_appendix.tex
TASK_TABLE=generated/task_contract_table.tex

.PHONY: all audit prompts amplification flash-directional proposal-diversity posthoc-checks logic-pruning clean

all: audit $(BUILD)/$(MAIN).pdf

audit:
	PYTHONPATH=../.. $(PYTHON) audit_contract_v2_readiness.py

prompts:
	PYTHONPATH=../.. $(PYTHON) ../../comparisons/llm_crn_generation/generate_contract_v2_prompt_review.py

amplification:
	PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.generate_local_logic_amplification
	PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.generate_hosted_portfolio_amplification
	PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.generate_portfolio_sensitivity_opportunities

flash-directional:
	PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.generate_flash_logic_directional

proposal-diversity:
	PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.analyze_llm_proposal_diversity

posthoc-checks:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) reviews/repo_answers_2026-09-09/analyze.py
	$(PYTHON) generate_posthoc_checks.py

logic-pruning:
	PYTHONPATH=../.. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl $(PYTHON) -m paper.iclr2027_genai_net_llm.analyze_logic_correctness_pruning --workers 8
	PYTHONPATH=../.. $(PYTHON) -m paper.iclr2027_genai_net_llm.generate_logic_pruning_table

$(BUILD)/$(MAIN).pdf: $(MAIN).tex macros.tex references.bib sections/*.tex figures/*.pdf $(PROMPT_APPENDIX) $(TASK_TABLE) generated/flash_logic_directional_tables.tex generated/llm_proposal_diversity_tables.tex $(wildcard generated/posthoc_*_table.tex) $(wildcard generated/logic_correctness_pruning_table.tex) generated/rpa_optimizer_tables.tex
	@mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	bibtex $(BUILD)/$(MAIN)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(MAIN).tex

clean:
	rm -f $(BUILD)/*.aux $(BUILD)/*.bbl $(BUILD)/*.blg $(BUILD)/*.log
	rm -f $(BUILD)/*.out $(BUILD)/*.pdf $(BUILD)/*.toc
