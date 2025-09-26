from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def publish_heads_mlflow(cfg, heads_dir: Path, metrics_json: Path,
                         oof_path: Path):
    mlflow.set_tracking_uri(cfg.registry.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.registry.mlflow.experiment)
    tags = {
        "task": "anorak_ratios",
        "encoder": cfg.encoder.name,
        "tile_size": str(cfg.prep.tile_size),
        "mag_label": cfg.mag_label,
        "cv": str(cfg.evaluation.tile_head.cv),
        "seed": str(cfg.run.seed),
    }
    with mlflow.start_run(tags=tags):
        mlflow.log_artifacts(str(heads_dir), artifact_path="heads")
        if Path(metrics_json).exists():
            mlflow.log_artifact(str(metrics_json), "metrics")
        if Path(oof_path).exists(): mlflow.log_artifact(str(oof_path), "preds")
        # optional: log params for reproducibility
        mlflow.log_params({
            "epochs":
            getattr(cfg.evaluation.tile_head.train, "epochs", 200),
            "lr":
            getattr(cfg.evaluation.tile_head.train, "lr", 5e-4),
            # ...
        })
        run_id = mlflow.active_run().info.run_id
    return run_id
