# Literature map for full-duplex simulator-in-the-loop search

This is a working map, not a substitute for a systematic review. Bibliographic
records included in `references.bib` were checked against primary publication
pages on 21 August 2026.

## Closest algorithmic precedents

- **OPRO: Large Language Models as Optimizers** (Yang et al., ICLR 2024).
  The closest precedent for showing an LLM previous solutions and objective
  values, then requesting improved candidates. Our distinction is concurrent RL,
  a persistent structured workspace, shared simulation accounting, and delayed
  provenance-aware merging.
- **Evolution through Large Models** (Lehman et al., 2022).
  Establishes an LLM as a mutation operator inside evolutionary search. This is
  the clearest precedent for the phrase "proposal operator."
- **FunSearch** (Romera-Paredes et al., Nature 2024).
  Connects LLM-generated programs to evaluator-driven selection and an evolving
  database. It motivates evaluator authority and retained high-performing state.
- **AlphaEvolve** (Novikov et al., 2025).
  Extends evaluator-guided LLM evolution to whole codebases with an evolutionary
  program database. This makes persistent evaluated memory established prior art;
  our distinction must rest on concurrent RL/LLM proposal channels and the CRN
  study, not the existence of a database.
- **Eureka** (Ma et al., 2023).
  Evolves batches of LLM-generated reward programs using statistics from RL
  policies trained in simulation. It is the closest LLM-plus-RL neighbor, but the
  LLM modifies the reward that trains RL rather than proposing candidate solutions
  alongside an independently learning RL search.
- **Reflexion** (Shinn et al., NeurIPS 2023).
  Uses external feedback and episodic textual memory without weight updates. It
  connects naturally to our readable reasoning notes and workspace memory.

## Scientific agents and grounded feedback

- **Scientific Generative Agent** (Ma et al., 2024).
  Couples LLM reasoning over discrete scientific hypotheses with simulation-based
  continuous optimization. It is the closest simulator-grounded scientific-agent
  comparison; our focus is asynchronous integration with an existing RL search.
- **Coscientist** (Boiko et al., Nature 2023).
  Demonstrates LLM orchestration of documentation, code, and laboratory tools,
  including optimization from prior experimental observations.
- **ChemCrow** (Bran et al., Nature Machine Intelligence 2024).
  Shows that chemistry-specific tools improve grounded task performance and also
  illustrates why fluent model output should not be treated as external evidence.
- **Self-driving laboratories** (Hase et al., Trends in Chemistry 2019) and
  **ChemOS** (Roch et al., PLOS ONE 2020) are useful broader connections if the
  paper develops the closed-loop scientific-discovery framing.

## Biomolecular design automation

- **Cello** (Nielsen et al., Science 2016) and **Cello 2.0** (Jones et al.,
  Nature Protocols 2022) anchor the design-automation lineage from behavioral
  specifications to experimentally realizable circuits.
- **CELLM** (Abello Castillo and Gutierrez Pescarmona, ACS Synthetic Biology
  2025) connects language models to Cello for natural-language circuit design and
  logical assistance. We therefore must not claim the first LLM system for genetic
  circuits; the narrower target is inverse dynamical CRN search with concurrent RL.
- **Robust genetic circuit design** (Schladt et al., ACS Synthetic Biology 2021)
  connects topology search, structural variants, parameter uncertainty, and
  robustness. It is particularly relevant to our topology-diversity analysis.
- The prior generative-RL framework should remain cited in the third person. ICLR
  permits citation of related author work and arXiv papers when handled this way.

## Systems and optimization connection

- **Asynchronous parallel Bayesian optimization** (Kandasamy et al., AISTATS
  2018) provides the cleanest established argument for avoiding global barriers
  under variable evaluation latency.
- Quality-diversity and island-model literature should be added once the final
  diversity estimand is frozen. MAP-Elites is a natural starting point, but the
  manuscript should avoid implying that the current method explicitly optimizes a
  quality-diversity objective.

## Positioning sentence

The defensible novelty claim is not that LLMs can propose or optimize candidates.
It is that a language-model proposal process and a learning-based optimizer can
exchange simulator-grounded state and structured candidates concurrently, through
an auditable persistent workspace, while sharing a fixed external evaluation
budget and preserving candidate provenance.
