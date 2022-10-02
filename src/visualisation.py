import pandas as pd
import matplotlib.pyplot as plt

x = []
y = []

path = "../results/res_firm_agent_run3.csv"
df = pd.read_csv(path)
output_plot = df.groupby("t")["output"].mean().plot()
plt.show()
age_plot = df.groupby("t")["age"].mean().plot()
plt.show()
total_effort_plot = df.groupby("t")["total_effort"].mean().plot()
plt.show()
n_workers_plot = df.groupby("t")["number_employees"].mean().plot()
plt.show()
n_firms_plot = df.groupby("t").size().plot()
plt.show()

