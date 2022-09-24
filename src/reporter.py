from src.base_model import BaseModel, Worker, Firm


class Reporter:

    def __init__(self, name, runid, out_dir, model):

        self.model : BaseModel = model
        self.step_idx = 0

        path_w = f"{out_dir}/res_worker_{name}_run{runid}.csv"
        self.out_w = open(path_w, "w")

        worker = ",".join(Worker.getResultsHeader())
        header = f"t,id,{worker}\n"
        self.out_w.write(header)

        path_f = f"{out_dir}/res_firm_{name}_run{runid}.csv"
        self.out_f = open(path_f, "w")

        firm = ",".join(Firm.getResultsHeader())
        header = f"t,id,{firm}\n"
        self.out_f.write(header)

    def close(self):
        self.out_w.close()
        self.out_f.close()

    def on_step(self):

        for a in self.model.schedule.agents:
            outFile = None

            resArr = a.getStepResults()
            resLine = ",".join(resArr)
            if a.type == "W":
                outFile = self.out_w
            elif a.type == "F":
                outFile= self.out_f

            outFile.write(f"{self.step_idx}, {a.unique_id}, {resLine}\n")

        self.step_idx += 1
