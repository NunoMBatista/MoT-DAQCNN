"""Read the Tissue topology-probe CSV and write the winning topologies into the
tissue_mnist capacity-sweep configs. Deterministic (pick max AUC), so it is safe
to run unattended in the overnight orchestrator."""
import csv, re

CSV = "outputs/paper_results/linear_probing/tissue_mnist_topology_probe.csv"
rows = list(csv.DictReader(open(CSV)))
singles = [r for r in rows if not str(r["topology"]).startswith("4k")]
best_z = max(singles, key=lambda r: float(r["auc_z"]) if r["auc_z"] else -1)["topology"]
best_zz = max(singles, key=lambda r: float(r["auc_zz"]))["topology"]
fourk = [r for r in rows if str(r["topology"]).startswith("4k")]
best4 = max(fourk, key=lambda r: float(r["auc_zz"]))
set4 = best4["topology"].split(":")[2].split("+")  # "4k:label:t1+t2+..."
print(f"best digital_z={best_z}  best digital_zz={best_zz}  best 4k={set4}")


def set_topo(path, topos):
    s = open(path).read()
    arr = "[" + ", ".join(f'"{t}"' for t in topos) + "]"
    s = re.sub(r"kernel_topology_names:.*", f"kernel_topology_names: {arr}", s)
    open(path, "w").write(s)


set_topo("configs/tissue_mnist/digital_z_best.yml", [best_z])
set_topo("configs/tissue_mnist/digital_zz_best.yml", [best_zz])
set_topo("configs/tissue_mnist/digital_zz_4k_best.yml", set4)
print("configs updated")
