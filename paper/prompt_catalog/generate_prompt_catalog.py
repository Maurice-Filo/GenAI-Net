#!/usr/bin/env python3
"""Generate the exact CRN Harness prompt catalog as a pdflatex document."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL4CRN.llm.benchmark_prompts import (  # noqa: E402
    CRN_AGENT_SYSTEM_PROMPT,
    get_reported_mmc2_task_prompt_2026,
)
from RL4CRN.llm.graphs import default_decider_writer_spec  # noqa: E402


TASKS = (
    ("logic", "Logic circuit"),
    ("rpa", "Robust perfect adaptation"),
    ("dose_hill", "Dose response: Hill"),
    ("dose_ultrasensitive", "Dose response: ultrasensitive"),
    ("dose_biphasic", "Dose response: biphasic"),
    ("classifier", "Autonomous classifier"),
    ("oscillator_mean", "Oscillator: temporal mean"),
    ("oscillator_frequency", "Oscillator: controlled frequency"),
    ("stochastic_rpa", "Stochastic robust perfect adaptation"),
)

REPORTED_PAPER_TASKS = TASKS[:-1]


def digest(text: str) -> str:
    return hashlib.sha256((text.rstrip() + "\n").encode("utf-8")).hexdigest()


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def prompt_block(text: str) -> str:
    return "\\begin{Prompt}\n" + text.rstrip() + "\n\\end{Prompt}\n"


def paper_prompt_block(text: str) -> str:
    return "\\begin{PaperPrompt}\n" + text.rstrip() + "\n\\end{PaperPrompt}\n"


def paper_digest_block(text: str) -> str:
    value = digest(text)
    return (
        "\\noindent\\textbf{SHA-256:} "
        f"\\texttt{{{value[:32]}}}\\\\\n"
        f"\\hspace*{{3.75em}}\\texttt{{{value[32:]}}}\n"
    )


def write_paper_appendix(
    rendered_tasks: list[tuple[str, str, str]], decider: str, writer: str
) -> Path:
    output = ROOT / "paper/iclr2027_genai_net_llm/generated/prompts_appendix.tex"
    parts = [
        r"""\section{Verbatim System and Task Prompts}
\label{app:verbatim-prompts}

This section reproduces the fixed textual contract shown to the Harness in the
reported deterministic experiments. Dynamic HOF, SIL, excluded-topology,
reaction-library, evaluator, and prior-request files are workspace state and
are therefore not duplicated here. Hashes use UTF-8 text with one terminal
newline. The Logic block is preserved exactly as executed, including its
final-state binary-cross-entropy surrogate; all candidates were scored by the
canonical trajectory evaluator described in Section~\ref{sec:experiments}.

\subsection{Shared operational system prompt}
""",
        paper_prompt_block(CRN_AGENT_SYSTEM_PROMPT),
        paper_digest_block(CRN_AGENT_SYSTEM_PROMPT),
        r"""
\clearpage
\subsection{Task-specific prompts}
""",
    ]
    for name, label, text in rendered_tasks:
        parts.extend(
            [
                f"\\subsubsection{{{latex_escape(label)}}}\n",
                f"\\textbf{{Task key:}} \\texttt{{{latex_escape(name)}}}\\\\\n",
                paper_digest_block(text),
                paper_prompt_block(text),
            ]
        )
    parts.extend(
        [
            r"""
\subsection{Operational graph prompts}

These two templates are shared across tasks. Braced fields are populated from
the immutable task prompt and current run workspace for each request.

\subsubsection{Decider prompt}
""",
            paper_prompt_block(decider),
            paper_digest_block(decider),
            r"""
\subsubsection{Writer prompt}
""",
            paper_prompt_block(writer),
            paper_digest_block(writer),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output


def main() -> None:
    output = Path(__file__).with_name("prompt_catalog.tex")
    spec = default_decider_writer_spec()
    decider = spec.get_node(spec.decider_node).prompt_template
    writer = spec.get_node(spec.writer_node).prompt_template
    rendered_tasks = [
        (name, label, get_reported_mmc2_task_prompt_2026(name, solver="CVODE"))
        for name, label in TASKS
    ]
    rendered_paper_tasks = [
        (name, label, get_reported_mmc2_task_prompt_2026(name, solver="CVODE"))
        for name, label in REPORTED_PAPER_TASKS
    ]

    parts = [
        r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=22mm]{geometry}
\usepackage{xcolor}
\usepackage{listings}
\usepackage[hidelinks]{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\definecolor{promptbg}{HTML}{F5F7F8}
\definecolor{promptframe}{HTML}{A8B2B8}
\lstnewenvironment{Prompt}{
  \lstset{
    basicstyle=\ttfamily\footnotesize,
    backgroundcolor=\color{promptbg},
    frame=single,
    rulecolor=\color{promptframe},
    breaklines=true,
    breakatwhitespace=true,
    columns=fullflexible,
    keepspaces=true,
    showstringspaces=false,
    aboveskip=7pt,
    belowskip=9pt
  }
}{}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{System and Task Prompts for the CRN Design Harness}
\author{Reproducibility Appendix}
\date{Frozen reported-campaign contract}
\begin{document}
\maketitle

\section{Prompt architecture}

Every experiment uses the same system prompt in Section~\ref{sec:system}.
The selected task contributes one task-specific block from
Section~\ref{sec:tasks}. The decider and writer templates in
Section~\ref{sec:graph} wrap that task text and dynamic run state.

Hall-of-Fame entries, SIL status, prior evaluator feedback, excluded
topologies, the indexed reaction library, and the requested candidate count
are dynamic context. They are intentionally not embedded in the fixed task
prompts. Logic and deterministic RPA are shown with the CVODE solver override
used by the reported campaigns. The Logic prompt is reproduced exactly as
executed; its stated final-state BCE objective was discovered during audit to
differ from the canonical transient-weighted L1 evaluator.

\section{Shared system prompt}\label{sec:system}
""",
        prompt_block(CRN_AGENT_SYSTEM_PROMPT),
        "\\noindent\\textbf{SHA-256:} \\nolinkurl{" + digest(CRN_AGENT_SYSTEM_PROMPT) + "}\n",
        r"""
\clearpage
\section{Task-specific prompts}\label{sec:tasks}
""",
    ]

    for name, label, text in rendered_tasks:
        parts.extend(
            [
                f"\\subsection{{{latex_escape(label)}}}\n",
                f"\\textbf{{Task key:}} \\texttt{{{latex_escape(name)}}}\\\\\n"
                f"\\textbf{{SHA-256:}} \\nolinkurl{{{digest(text)}}}\n",
                prompt_block(text),
            ]
        )

    parts.extend(
        [
            r"""
\clearpage
\section{Operational graph templates}\label{sec:graph}

These templates are shared across tasks. Braced fields are populated at each
request from the task prompt and current run workspace.

\subsection{Decider template}
""",
            prompt_block(decider),
            "\\noindent\\textbf{SHA-256:} \\nolinkurl{" + digest(decider) + "}\n",
            r"""
\subsection{Writer template}
""",
            prompt_block(writer),
            "\\noindent\\textbf{SHA-256:} \\nolinkurl{" + digest(writer) + "}\n",
            r"""
\section{Task index and runtime solver}

\begin{longtable}{@{}lll@{}}
\toprule
Task key & Task family & Runtime evaluator \\
\midrule
\endhead
logic & Boolean logic & CVODE \\
rpa & Deterministic tracking & CVODE \\
dose\_hill & Dose response & CVODE \\
dose\_ultrasensitive & Dose response & CVODE \\
dose\_biphasic & Dose response & CVODE \\
classifier & Autonomous classification & CVODE \\
oscillator\_mean & Autonomous oscillation & CVODE \\
oscillator\_frequency & Input-controlled oscillation & CVODE \\
stochastic\_rpa & Stochastic tracking and noise & GPU SSA \\
\bottomrule
\end{longtable}

\end{document}
""",
        ]
    )
    output.write_text("".join(parts), encoding="utf-8")
    print(output)
    print(write_paper_appendix(rendered_paper_tasks, decider, writer))


if __name__ == "__main__":
    main()
