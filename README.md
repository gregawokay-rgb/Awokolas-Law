# Awokola's Law of Attention Dynamics

A phenomenological and heuristic framework for modeling information retrieval and attention degradation within the context windows of causal autoregressive large language models.

## The Mathematical Formulation

Awokola's Law models the empirical "lost in the middle" phenomenon as a continuous negative exponential distribution, mapping information retention probability against relative token position, parameter scale, and textual distraction factors:

\[A(x, P, D) = \exp\left( -\frac{\gamma}{D \cdot \ln(P) \cdot [1 + 4(x - 0.5)^2]} \right)\]

### Core Variables & System Parameters

*   **A (Attention Retrieval Probability):** The mathematical likelihood (0 to 1) of successful token synthesis and extraction during generation.
*   **x (Normalized Token Position):** The relative coordinate of information location within the prompt context window (x=0 is the start, x=0.5 is the absolute sequence midpoint, x=1 is the terminal end).
*   **P (Model Parameter Scale):** The total dense computational capacity and parameter count of the neural network.
*   **D (Context Quality & Formatting Density):** A structural coefficient tracking prompt noise. Lower values indicate clean, well-formatted text; higher values represent repetitive or chaotic text data.
*   **γ (Systemic Fatigue Constant):** An intrinsic factor mapping transformer head scaling and positional embedding overhead.

---

## Access the Full Academic Paper

The complete typeset academic layout detailing the boundary proofs, scale amortization limits, and engineering applications can be found in the repository files:
*   [Download the PDF Paper](./Awokolas_Law.pdf)
*   [View the LaTeX Source Code](./main.tex)

## Interactive Simulator

You can use the included `simulator.py` script to generate local visualization plots of Awokola's Law across various model capacities (such as 7B, 70B, and 405B parameters).

## Citation

If you utilize this heuristic framework or curve projection in your own LLM context evaluation or prompt engineering research, please cite this repository:

```text
Awokola, O. (2026). Awokola's Law: A Heuristic Decay Profile for Context Window Attention Loss in Large Language Models. GitHub Repository: https://github.com
```
