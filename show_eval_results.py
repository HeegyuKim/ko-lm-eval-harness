import pandas as pd
from glob import glob
import json
import os


files = glob("output/*.json")
dfs = []

for file in files:
    with open(file) as f:
        js = json.load(f)
    df = pd.DataFrame(js["results"]).transpose().reset_index()
    df["task"] = df["index"].map(lambda x: "_".join(x.split("_")[:-1]))
    df["filename"] = os.path.splitext(os.path.basename(file))[0]

    dfs.append(df)

df = pd.concat(dfs, axis=0)
scores = df.groupby(["task", "filename",]).mean(True).reset_index()
print(scores.to_markdown(index=False))