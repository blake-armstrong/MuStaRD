import numpy as np
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], header=None)
df.columns = ["step"]
step = 0
diffs = []
for index, row in df.iterrows():
    diffs.append(row["step"] - step)
    step = row["step"]
# diffs.append(1500000 - step)
diffs = np.array(diffs)
# print(len(df.index))
# print((len(df.index)) / (1.5 * 1e-9))
print("Differences: ", diffs)
minimum_time = 100
diffs = diffs[diffs > minimum_time]
print(f"Differences > {minimum_time} fs: ", diffs)
avg_diffs = np.mean(diffs)
print("Average difference: ", avg_diffs)
print("Average difference (seconds): ", avg_diffs * 1e-15)
print("Average difference (seconds): ", 1 / (avg_diffs * 1e-15))
print("Diffusion coefficient: ", (1 / 6) * (2.9 * 1e-8) ** 2 * 1 / (avg_diffs * 1e-15))
