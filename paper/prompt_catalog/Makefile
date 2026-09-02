PYTHON ?= python3
PDFLATEX ?= pdflatex
export MPLCONFIGDIR ?= /local0/tmp/mpl
export TEXMFVAR ?= /local0/tmp/texmf-var

.PHONY: all clean

all: prompt_catalog.pdf

prompt_catalog.tex: generate_prompt_catalog.py ../../RL4CRN/llm/benchmark_prompts.py ../../RL4CRN/llm/graphs.py
	$(PYTHON) generate_prompt_catalog.py

prompt_catalog.pdf: prompt_catalog.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error prompt_catalog.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error prompt_catalog.tex

clean:
	rm -f prompt_catalog.aux prompt_catalog.log prompt_catalog.out prompt_catalog.pdf prompt_catalog.tex
