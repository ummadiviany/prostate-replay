import wandb
api = wandb.Api()
run = api.run("vinayu/CL_Replay/17hrwaha")
import pandas as pd

# print(run.summary)

# print(run.config)

# logged_metrics = run.history()
# logged_metrics.to_csv('wandb_metrics.csv')
print(f"Run name : {run.name}")
