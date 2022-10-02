from src.base_model import BaseModel, Worker, Firm


class Reporter:

    def __init__(self, name, runid, out_dir, model, attributes_worker_tuple, attributes_firm_tuple):

        self.model: BaseModel = model
        self.step_idx = 0

        path_w = f"{out_dir}/res_worker_{name}_run{runid}.csv"
        self.out_w = open(path_w, "w")

        worker = ",".join(attributes_worker_tuple)
        header = f"t,id,{worker}\n"
        self.out_w.write(header)

        path_f = f"{out_dir}/res_firm_{name}_run{runid}.csv"
        self.out_f = open(path_f, "w")

        firm = ",".join(attributes_firm_tuple)
        header = f"t,id,{firm}\n"
        self.out_f.write(header)

    def close(self):
        self.out_w.close()
        self.out_f.close()

    def on_step(self, attributes_worker_tuple, attributes_firm_tuple):

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

            outFile.write(f"{self.step_idx}, {a.unique_id}, {resLine}\n")

        self.step_idx += 1
