from pathlib import Path
import wandb

run = wandb.init(project="collection-linking-quickstart")

artifact_filepath = Path("./my_model_artifact.txt")
artifact_filepath.write_text("simulated model file")

logged_artifact = run.log_artifact(artifact_filepath, "artifact-name", type="model")
run.link_artifact(
    artifact=logged_artifact,
    target_path="upasanapaul2030-technical-university-of-denmark-org/wandb-registry-model/correupt_mnist",
)
run.finish()
