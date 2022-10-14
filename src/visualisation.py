import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd


class Visualisation:

    def __init__(self, name, runid, out_dir, plot_style):

        path_w = f"{out_dir}/res_worker_{name}_run{runid}.csv.gz"
        path_f = f"{out_dir}/res_firm_{name}_run{runid}.csv.gz"
        plt.style.use(f"{plot_style}")

        worker_dk = dd.read_csv(path_w, blocksize=4000000, compression="gzip")
        firm_dk = dd.read_csv(path_f, blocksize=4000000, compression="gzip")

        firm_age_group = firm_dk.groupby(["age"])["number_employees", "output"].mean()
        fa_df = firm_age_group.compute()
        plt.figure(figsize=(12, 8))
        fa_df.unstack().T.sum().plot()
        plt.title("Evolution of Workers/Output over firm lifetime")
        plt.show()

#df = pd.read_csv(path)
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
