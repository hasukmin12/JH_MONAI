import wandb
wandb.init(project="has_kipa", entity="hutom_miai")
loss = 0.5
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
wandb.log({"loss": loss})