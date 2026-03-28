import torch
from src.layers.daqk import DAQKLayer

layer = DAQKLayer(
    n_qubits=9,
    grid_size=3,
    kernel_topology_names=["kings"],
    mode="trotter",
    evolution_time=2.5,
    include_correlators=True,
    encoding_mode="analog",
    interface="torch"
)

x = torch.rand(4, 9) # batch of 4
try:
    out = layer(x)
    print("Success! Output shape:", out.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
