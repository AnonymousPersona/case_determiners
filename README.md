## Reproducibility

This repository contains all data, models, and scripts required to reproduce the analyses reported in the manuscript. All file paths are project-relative and resolved using `here::here()`.

### System requirements
- R (≥ 4.2 recommended)
- A C++ toolchain compatible with CmdStan
- CmdStan (via the `cmdstanr` R package)
- Sufficient memory/CPU to run Bayesian models (parallel chains are enabled)

### Reproducing the analyses

1. Clone the repository and enter it:

```
git clone https://github.com/AnonymousPersona/case_determiners
cd case_determiners
```

2.	Ensure CmdStan is installed (one-time setup):

```
cmdstanr::install_cmdstan(cores = parallel::detectCores())
```

3.	Run the full analysis pipeline:

```
Rscript run_all.R
```
