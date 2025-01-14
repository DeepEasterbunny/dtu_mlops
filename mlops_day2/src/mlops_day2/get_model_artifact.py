import wandb
import torch
from model import MyAwesomeModel
from evaluate import evaluate_model
if __name__ == "__main__":
   REGISTRY = 'model'
   COLLECTION = 'MLOps'
   ALIAS = 'latest'

   run = wandb.init(
      entity = 's203768-dtu',
      project = 'MLOps'
   )  

   artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

   fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
   # Access and download model. Returns path to downloaded artifact
   downloaded_model_path = run.use_model(name= artifact_name)
   model = MyAwesomeModel()
   model.load_state_dict(torch.load("models/A_trained_model.ckpt"))
   # Random 1,1,28 ,28 tensor
   noise = torch.randn(1, 1, 28,28)
   # Evaluate model
   model.eval()
   with torch.no_grad():
      output = model(noise)
      print(output)
      

