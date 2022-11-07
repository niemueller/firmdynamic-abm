# Agent-based Model (ABM) Implementation of Axtell's (99) Endogenous Firm Dynamics Model

This repository contains the implementation of the agent-based model used to model endogenous firm dynamics and labor flows via heterogeneous agents.
The original model was developed by Axtell in 1999 and a recently adapted in 2018. Applied to US data it was able to reproduce a variety of cross-secitonal properties of U.S
businesses. In the context of a master thesis, an attempt was made to reproduce this model and to replicate the resulting findings.


## Running the model

1. The core model implementation can be found in [`src/base_model_async_random.py`](src/base_model_async_random.py).

3. A script in [`scripts/run-base_axtell_99.py`](scripts/run-base_axtell_99.py) can be configured to identify which scenarios will be simulated and basic parameters such as firm production parameters `a,b & beta` can be defined.

4. Visualisation for the replication are done in [`scripts/axtell-99-vis.ipynb`](scripts/axtell-99-vis.ipynb)
