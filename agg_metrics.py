import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from tabulate import tabulate

api = wandb.Api()
necessay_keys = ["4_prostate158_curr_dice", "4_isbi_curr_dice", "4_promise12_curr_dice", "4_decathlon_curr_dice",
                 "Accuracy (ACC)", "Backward Transfer (BWT)", "Average Forgetting (AFGT)"]


# function to get the run summary metrics from wandb api
def get_run_summary_metrics(run):
    
    run_summary_data = {k: v for k, v in run.summary.items() if k in necessay_keys}
    run_summary_data = {k: run_summary_data[k] for k in necessay_keys}
    return run_summary_data



if __name__ == "__main__":
    # entity, project = "vinayu", "CL_Replay"
    # runs = api.runs(entity + "/" + project) 
    df = pd.DataFrame()
    
    # run_ids = ["vinayu/CL_Replay/3bm8mcn3",
    #            "vinayu/CL_Replay/103t1sa3",
    #            "vinayu/CL_Replay/1i3j20wj"]
    # run_ids = [
    #     "vinayu/CL_Replay/1uyl3o8x",
    #     "vinayu/CL_Replay/222vq4yc",
    #     "vinayu/CL_Replay/3rzr8ibf",
    # ]
    # run_ids = [
    #     "vinayu/CL_Replay/makm2e76",
    #     "vinayu/CL_Replay/38stqirz",
    #     "vinayu/CL_Replay/2dl8aomm"
    # ]
    # run_ids = [
    #     "vinayu/CL_Replay/2md387fn",
    #     "vinayu/CL_Replay/2acs8a4w",
    #     "vinayu/CL_Replay/m4ftg1ow"
    # ]
    
    # run_ids = [
    #     "vinayu/CL_Replay/12abd1qx",
    #     "vinayu/CL_Replay/379li9rr",
    #     "vinayu/CL_Replay/hie84o0d",
    #     "vinayu/CL_Replay/qfa8kbwm",
    # ]
    
    run_ids = [
        "vinayu/CL_Replay/12ih1pcu",
        "vinayu/CL_Replay/1hnz204t",
        "vinayu/CL_Replay/2mt7zuig",
        "vinayu/CL_Replay/setcf7ek"
    ]
    runs = [api.run(run_id) for run_id in run_ids]
    
    for run in runs:
        print(f"Run name: {run.name}")
        run_data = get_run_summary_metrics(run)
        run_data["run_name"] = run.name
        run_data["run_id"] = run.id
        # df = df.append(run_data, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(run_data, index=[0])], ignore_index=True)
        
    filename = '__'.join([run.name for run in runs])
    df.to_csv(f"summary_csvs/{filename}.csv", index=False)
    
    
    means = df.mean(axis=0, numeric_only=True)
    stds = df.std(axis=0, numeric_only=True)
    pm = "\u00B1"
    rows = [f"{round(m,2)} {pm} {round(s,2)}" for m, s in zip(means.values, stds.values)]
    print('\n'*2)
    print(tabulate([rows], headers=means.index, tablefmt="fancy_grid"))
    print('\n'*2)