# BOP-Elites dev run instructions
1. Clone this repo.
1. Start a virtual environment and install dependencies.
   ```bash
   python3.9 -m virtualenv pyribs_BOP-Elites
   cd pyribs_BOP-Elites
   source bin/activate
   pip install -e .[all,dev]  # `all` contains dependencies for all extras of pyribs.
                              # `dev` contains development dependencies.
   ```
1. We are using wandb to log metrics. Run `wandb login` and paste in the API key from your Account->User Settings->API keys.
1. Run an experiment with `python boe_test.py`.
1. Passive archive heatmaps and algorithm checkpoints are written to `test_logs`.
1. If you need to reload from a checkpoint, find a variable named `reload_checkpoint` from within `boe_test.py` (should be around L103) and set it to the path of a checkpoint within `test_logs`.