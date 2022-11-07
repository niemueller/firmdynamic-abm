import gzip
from src.base_model_grid import BaseModel


class Reporter:

    def __init__(self,
                 name,
                 runid,
                 out_dir,
                 model,
                 attributes_worker_tuple,
                 attributes_firm_tuple,
                 attributes_model_tuple,
                 optimization_type):

        self.model: BaseModel = model
        self.step_idx = 0

        path_w = f"{out_dir}/res_worker_{name}_run{runid}_opttype{optimization_type}.csv.gz"
        self.out_w = gzip.open(path_w, "wt")

        worker = ",".join(attributes_worker_tuple)
        header = f"t,id,{worker}\n"
        self.out_w.write(header)

        path_f = f"{out_dir}/res_firm_{name}_run{runid}_opttype{optimization_type}.csv.gz"
        self.out_f = gzip.open(path_f, "wt")

        firm = ",".join(attributes_firm_tuple)
        header = f"t,id,{firm}\n"
        self.out_f.write(header)

        path_m = f"{out_dir}/res_model_{name}_run{runid}_opttype{optimization_type}.csv.gz"
        self.out_m = gzip.open(path_m, "wt")

        model = ",".join(attributes_model_tuple)
        header = f"t,{model}\n"
        self.out_m.write(header)

    def close(self):
        self.out_w.close()
        self.out_f.close()
        self.out_m.close()

    def on_step(self, attributes_worker_tuple, attributes_firm_tuple, attributes_model_tuple):

        for a in self.model.schedule.agents:
            outFile = None

            if a.type == "W":
                resArr = a.getStepResults(attributes_worker_tuple)
                resLine = ",".join(str(v) for v in resArr)
                outFile = self.out_w
            elif a.type == "F":
                resArr = a.getStepResults(attributes_firm_tuple)
                resLine = ",".join(str(v) for v in resArr)
                outFile = self.out_f

            outFile.write(f"{self.step_idx},{a.unique_id},{resLine}\n")

        outFile = None
        resArr = self.model.getStepResults(attributes_model_tuple)
        resLine = ",".join(str(v) for v in resArr)
        outFile = self.out_m
        outFile.write(f"{self.step_idx},{resLine}\n")

        self.step_idx += 1
