# MDP Policy Pursuit

An agent on an 11x11 grid tries to catch two crew members while avoiding a moving alien. The whole thing is modeled as a Markov Decision Process with about 11000 reachable states and solved with policy iteration (gamma=0.95). The crew members have their own behavior: they flee when the agent gets within 2 cells and move randomly otherwise. Once the optimal policy is computed, two neural networks (one in Julia with Flux, one in Python with PyTorch) are trained to approximate it, so the agent can act without needing the full state table.

`Proj3.jl` builds the full state space (agent position x crew1 x crew2 x alien position), constructs transition matrices for each of the five actions (four directions + stay), and runs policy iteration until convergence. Rescue gives +200 reward, getting caught by the alien gives -1000.

`Proj3_Bonus.jl` takes the converged policy and trains a Flux neural network to approximate it.

`train_general.py` does the same thing in PyTorch with a three-layer network. Both convert each state into a feature vector (positions, distances, relative directions) and predict action probabilities.

Julia notebooks:

```bash
julia -e 'using Pkg; Pkg.add(["Pluto", "Flux", "Statistics", "LinearAlgebra"])'
julia -e 'using Pluto; Pluto.run()'
```

Open `Proj3.jl` or `Proj3_Bonus.jl` from the Pluto interface.

Python version:

```bash
pip install torch numpy
python train_general.py
```
