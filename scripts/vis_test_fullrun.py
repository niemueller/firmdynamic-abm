import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd


optimization_type = 4
out_dir = "../results"
name = "agent"
runid = "5"
plot_style = "fivethirtyeight"

path_f = f"{out_dir}/res_firm_agent_run1.csv.gz"
#path_f = f"{out_dir}/res_firm_{name}_run{runid}_opttype{optimization_type}.csv.gz"
plt.style.use(f"{plot_style}")

#worker_dk = dd.read_csv(path_w, compression="gzip")
firm_dk = dd.read_csv(path_f, compression="gzip")

# plot number of firms over time
firm_number = firm_dk.groupby(["t"])["t"].size()
firm_number_df = firm_number.compute()
plt.figure(figsize=(12, 8))
firm_number_df.plot()
plt.xlabel("Model step")
plt.ylabel("Number of firms")
plt.title("Number of firms over time")
plt.savefig(f"{out_dir}/fullrun_firm_numbfirms.png")
plt.show()

#
# # Plot Evolution of the number of workers and output over the lifetime of a firm
# print(firm_dk.head(n=100))
# firm_age_group = firm_dk.groupby(["age"])["number_employees", "output"].mean()
# fa_df = firm_age_group.compute()
# plt.figure(figsize=(12, 8))
# fa_df.plot()
# plt.title("Evolution of Workers/Output over firm lifetime")
# plt.savefig(f"{out_dir}/{runid}_worker_output_evo.png")
# plt.show()
#
#
# # Plot Evolution of the number of workers and output over lifetime of a firm for 10 firms with average lifetime
# average_age = firm_dk.groupby("id")["age"].max()
# avg_age_df = average_age.compute()
# plt.figure(figsize=(12, 8))
# print(avg_age_df)
# avg_age = avg_age_df.mean()
# print(avg_age)  # 36 years (company with id = 10 exactly 36 years old
# firm10 = firm_dk[firm_dk.id == 10].compute()
# plt.figure(figsize=(12, 8))
# firm10.plot(x="age", y="number_employees")
# plt.title("Evolution of Workers/Output over firm lifetime for Company ID10")
# plt.savefig(f"{out_dir}/{runid}_worker_output_evo_onecomp.png")
# plt.show()
#
#
# avg_age_df.plot()
# plt.title("Average Age")
# plt.show()
