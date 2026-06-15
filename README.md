# Design of Sparse Control for Network Opinion Manipulation

Recovering piecewise-sparse control inputs that steer a 25-node opinion network to target final states under a per-step actuation budget, treated as a compressed-sensing recovery problem.

## Project context

This is a project for **EE4740 Data Compression: Entropy and Sparsity Perspectives** at TU Delft. The assignment is to design and study sparse control for network opinion manipulation: an external actor influences a few nodes of a social network at each time step and tries to drive the network to a desired opinion profile, and the task is to find how little influence, placed where and when, suffices. The control problem is reframed as piecewise-sparse signal recovery and analysed with the sparsity and recovery tools of the course.

Authors:

- Adam El Haddouchi (5476526), MSc Electrical Engineering, TU Delft
- Ilyaas Shousha (5593247), MSc Electrical Engineering, TU Delft

## The problem

The network has 25 nodes with opinion vector $x_k$ evolving under linear dynamics

$$x_{k+1} = A x_k + u_k, \qquad k = 0, \dots, N-1, \quad N = 25$$

where $u_k$ is the external control applied at step $k$ and the input matrix is the identity, so the manipulator can in principle address any node. Unrolling the recursion over the horizon gives $x_N = A^N x_0 + \sum_k A^{N-k} u_k$. Writing $b = x_N - A^N x_0$ and stacking the per-step controls as $u = [u_1; \dots; u_N]$ turns the schedule into a single linear system

$$b = \Phi u, \qquad \Phi = [\, A^{N-1} \mid A^{N-2} \mid \dots \mid A \mid I \,] \in \mathbb{R}^{25 \times 625}$$

The stacked control $u$ is **piecewise sparse**: it splits into 25 blocks of length 25, one block per time step, and the per-step budget $s$ caps the number of nonzeros inside each block. This is not block sparsity. Every block is allowed to be active, and only the count of active nodes within a step is limited, so a method that turns whole blocks on or off has the wrong inductive bias.

The dataset is `data/PiecewiseSparse.mat`, with two arrays: $A$ ($25 \times 25$) and `FinalState` ($1000 \times 25$), giving 1000 target final states. $A$ is dense and signed, not symmetric and not row-stochastic, with spectral radius about 6.08 and complex eigenvalues. With only $A$ and the final states supplied, the initial state is taken as $x_0 = 0$, so each row of `FinalState` is a target $b$.

Why it matters: sparse actuation is limited-budget influence, the kind that arises in marketing or opinion shaping. The practical question, how little influence suffices to move a network and where it should be placed, is exactly the compressed-sensing and sparsity track of the course applied to a control problem.

## Methods

All solvers run on the unit-normalized dictionary and rescale their coefficients back to physical units before any energy or residual is reported, so results across methods are comparable.

- **OMP (orthogonal matching pursuit)** — structure-blind greedy recovery. It serves as the baseline that isolates the value of knowing the block structure (Tropp and Gilbert, 2007).
- **POMP (piecewise OMP)** — the structured method: the OMP loop on the single shared system with a per-block counter, abandoning a block once it reaches its budget $s$. It is the only method that enforces the per-step actuation budget by construction (Li et al., 2016).
- **$\ell_1$ / basis pursuit (with BPDN for the noisy case)** — the convex baseline, $\min \|u\|_1$ subject to $\Phi u = b$. It is a different algorithm class with a different failure mode, and it cannot be told a per-block budget (Candes et al., 2006).
- **Oracle least squares** — least squares on the true support. It is a benchmark, not a runnable method: it lower-bounds achievable energy and upper-bounds achievable recovery.

Group-LASSO and other whole-block selectors are deliberately excluded, because they model block sparsity (whole blocks on or off) while here every block is active and only the within-block count is capped.

## Key findings

- **Conditioning is the load-bearing numerical issue.** Because $A$ has spectral radius about 6.08, the column scales of $\Phi$ span roughly eighteen orders of magnitude, from about $2.55 \times 10^{18}$ in the earliest block ($A^{24}$) down to $1.0$ in the identity block, and the raw dictionary has condition number about $5.14 \times 10^{16}$ (numerically singular). Normalizing the columns to unit norm brings the condition number to $16.67$. Without normalization the first greedy atom lands in the earliest block for all 100 sampled targets, with correlations about $5 \times 10^{17}$ times larger there purely by column scale; after normalization the atom is chosen on relevance instead.
- **Feasibility is essentially free.** With 25 equations and up to 25 nonzeros, even a budget of one active node per step ($s = 1$) reaches any target, with a median reconstruction residual of about $6.7 \times 10^{-10}$. The budget is therefore an energy and timing knob, not a feasibility knob.
- **Sparsity-energy tradeoff.** Median control energy under POMP falls from $7.96$ at $s = 1$ to $4.71$ at $s = 5$, converging toward the budget-independent OMP energy of $4.28$. Basis pursuit is lower ($3.93$) but ignores the per-step budget, so its schedule is not admissible. Unconstrained, the control concentrates in the final steps; a tighter budget forces it to act earlier, when its influence on the final state is weaker.
- **The value of the structure is feasibility, not recovery.** POMP is the only method that returns a budget-admissible control; OMP and basis pursuit pile up to 17 atoms into a single step. On exact support recovery, POMP ties OMP rather than beating it.
- **Recovery is governed by the number of active inputs.** On synthetic known-support problems, the probability of exact support recovery shows a clean phase transition in the number of active blocks $L$ (the active-input count), crossing one half between $L = 4$ and $L = 5$ and falling to near zero by $L = 8$. The per-step budget $s$ does not move recovery, and noise lowers all methods together. This matches the sparse-controllability prediction that the minimum number of active inputs governs identifiability (Joseph and Murthy, 2021).
- **The supplied targets carry no unique support.** $A$ is invertible (determinant about $-1.23 \times 10^{12}$), so every single block is itself a basis for the 25-dimensional state space, and unconstrained recovery collapses about 78% of its atoms onto the final identity block. Support recovery is therefore measured on synthetic problems with a planted support rather than on the given targets.

The study runs in two experiment modes: Mode A uses the 1000 supplied targets to measure energy, residual, runtime, and the per-step energy profile as $s$ varies, and Mode B draws synthetic controls with known support to measure exact support recovery against the number of active blocks and against the signal-to-noise ratio.

## Repository structure

```
network-opinion-control/
├── data/
│   └── PiecewiseSparse.mat        # A (25x25), FinalState (1000x25)
├── notebooks/
│   └── Notebook.ipynb             # full analysis: all methods and experiments
├── figures/                       # all generated figures (PNG)
│   ├── phi_column_scales.png
│   ├── energy_vs_s.png
│   ├── residual_vs_s.png
│   ├── runtime_vs_s.png
│   ├── per_step_energy.png
│   ├── support_recovery_vs_L.png
│   ├── recovery_vs_snr.png
│   ├── method_comparison.png
│   ├── support_block_distribution.png
│   ├── l1_per_block_budget_violation.png
│   └── pomp_per_block_budget.png
├── requirements.txt
└── README.md
```

## Setup and reproduction

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset `data/PiecewiseSparse.mat` is included in the repository, so no download is needed.

To reproduce the analysis, open the notebook in Jupyter, select the **Python (network-opinion-control)** kernel, and run all cells:

```bash
jupyter notebook notebooks/Notebook.ipynb
```

To run it non-interactively from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Notebook.ipynb
```

All random seeds are fixed (`numpy.random.default_rng(42)`), so the Monte Carlo experiments and every reported number reproduce exactly on a clean run.

## Requirements

The main packages (see `requirements.txt` for exact version pins) are:

- **numpy** and **scipy** — numerical arrays, linear algebra, and loading the `.mat` dataset
- **matplotlib** — all figures
- **cvxpy** (with the CLARABEL solver) — the convex basis pursuit and BPDN solves
- **pandas** — the method-comparison table
- **jupyter**, **ipykernel**, and **notebook** — running the notebook

## Authors

- Adam El Haddouchi (5476526), MSc Electrical Engineering, TU Delft
- Ilyaas Shousha (5593247), MSc Electrical Engineering, TU Delft

## Acknowledgments

Project for the EE4740 Data Compression course at TU Delft. Thanks to the course consultant Geethu Joseph, whose work on controllability of linear systems under input sparsity and on network opinion control provides the theoretical anchor for the recovery results, and with whom the $x_0 = 0$ modelling assumption is to be confirmed.
