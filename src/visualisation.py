import pandas as pd
import matplotlib.pyplot as plt

x = []
y = []

path = "/home/debian/abm/results/res_firm_agent_run5.csv"
df = pd.read_csv(path)
output_plot = df.groupby("t")["output"].mean().plot()
plt.title("Firm Output")
plt.ylabel("output")
plt.xlabel("time")
plt.show()
age_plot = df.groupby("t")["age"].mean().plot()
plt.title("Firm Age")
plt.show()
total_effort_plot = df.groupby("t")["total_effort"].mean().plot()
plt.title("Average Total Effort")
plt.show()
n_workers_plot = df.groupby("t")["number_employees"].mean().plot()
plt.title("Average Number of Employees")
plt.xlabel("time")
plt.ylabel("Average Firm Size")
plt.show()
n_firms_plot = df.groupby("t").size().plot()
plt.title("Number of Firms over Time")
plt.xlabel("time")
plt.ylabel("Number of firms")
plt.show()
