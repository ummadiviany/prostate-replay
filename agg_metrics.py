import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from tabulate import tabulate

api = wandb.Api()
necessary_keys = ["4_prostate158_curr_dice", "4_isbi_curr_dice", "4_promise12_curr_dice", "4_decathlon_curr_dice",
                 "Accuracy (ACC)", "Backward Transfer (BWT)", "Average Forgetting (AFGT)"]

joint_necessary_keys = ["prostate158_curr_dice", "isbi_curr_dice", "promise12_curr_dice", "decathlon_curr_dice",]

# function to get the run summary metrics from wandb api
def get_run_summary_metrics(run, necessary_keys):
    
    run_summary_data = {k: v for k, v in run.summary.items() if k in necessary_keys}
    run_summary_data = {k: run_summary_data[k] for k in necessary_keys}
    return run_summary_data



if __name__ == "__main__":
    # entity, project = "vinayu", "CL_Replay"
    # runs = api.runs(entity + "/" + project) 
    df = pd.DataFrame()
    
    # Representative runs (linspace) Worst case
    # run_ids = [
    #     "vinayu/CL_Replay/12abd1qx",
    #     "vinayu/CL_Replay/379li9rr",
    #     "vinayu/CL_Replay/hie84o0d",
    #     "vinayu/CL_Replay/qfa8kbwm",
    # ]
    
    # Representative runs (linspace) Practical case
    # run_ids = [
    #     "vinayu/CL_Replay/12ih1pcu",
    #     "vinayu/CL_Replay/1hnz204t",
    #     "vinayu/CL_Replay/2mt7zuig",
    #     "vinayu/CL_Replay/setcf7ek"
    # ]
    
    # Imprtance sampling runs Worst case
    # run_ids = ["vz3xek4r", "uquflia2", "1se1h3df", "1587x1vi"]
    # Importance sampling runs Practical case
    # run_ids = ["1w83y29a", "1t18fpg7", "1kefi6az", "18gj2sfo"]
    
    # Sequential (Adam)
    # run_ids = ["1eflar5w", "1v1meejw", "2h3e3q3n", "2x951dcn"]
    # run_ids = ["1ux12qb9", "1hs38hhl", "19wm9gxo", "185djjvl"]
    
    # Sequential (SGD)
    # run_ids = ["33ec8dw4", "xpp035m1", "2d625md3", "2esu6ugy"]
    # run_ids = ["1ebk7d28", "24oj11da", "3jrgl9tu", "11w1z8e5"]
    
    # Joint(100)
    # run_ids = ["1t9mo8og", "p9mvl9di", "1tv564ms", "m025f288"]
    # Joint (50)
    # run_ids = ["1bzlueau", "28dc3a1r", "1n43di84", "1c7k94gq"]
    
    # Sequential (SGD) (100)
    # run_ids = ["162u5r49", "29tkcwbw", "crbuunhn", "d7xyq3yl"]
    # run_ids = ["233iyzsk", "2fcstdwp", "4pw7rewn", "1lss1kux"]
    
    # Sequential (SGD) (50)
    
    # Random Sampling runs 
    # Worst case
    # run_ids = ["1xpusaof", "298foogv", "2u9d54o7", "2u66xy2s"]
    # Practical case
    run_ids = ["1e98rhnh", "3ljzmerp", "lje1d3uf", "mkfb1xjf"]
        
    runs = [api.run(f"vinayu/CL_Replay/{run_id}") for run_id in run_ids]
    for run in runs:
        print(f"Run name: {run.name}")
        run_data = get_run_summary_metrics(run, necessary_keys)
        run_data["run_name"] = run.name
        run_data["run_id"] = run.id
        # df = df.append(run_data, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(run_data, index=[0])], ignore_index=True)
        
    filename = '__'.join([run.name for run in runs])
    df.to_csv(f"summary_csvs/{filename}.csv", index=False)
    
    
    means = df.mean(axis=0, numeric_only=True)
    stds = df.std(axis=0, numeric_only=True)
    pm = "\u00B1"
    rows = [f"{round(m,1)} {pm} {round(s,1)}" for m, s in zip(means.values, stds.values)]
    print('\n'*2)
    print(tabulate([rows], headers=means.index, tablefmt="github"))
    print('\n'*2)