import wandb
import torch

try:
    wandb.init(project="test-connection", name="connection-check")
    print("✅ WandB Connection Successful!")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    wandb.log({"test_metric": 1.0})
    wandb.finish()
except Exception as e:
    print(f"❌ Connection Failed: {e}")