# Lightweight logging adapter: mlflow | wandb | none
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class BaseLogger:

    def __init__(self, cfg):
        self.cfg = cfg

    def start(self, run_name: str, tags: Dict[str, str]):
        pass

    def log_params(self, params: Dict[str, object]):
        pass

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step: Optional[int] = None):
        pass

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None):
        pass

    def log_artifacts(self,
                      dir_path: Path,
                      artifact_path: Optional[str] = None):
        pass

    def end(self):
        pass


class NullLogger(BaseLogger):
    pass  # all no-ops


class MLflowLogger(BaseLogger):

    def __init__(self, cfg):
        super().__init__(cfg)
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        self.mlflow.set_experiment(cfg.logger.project)
        self._active = False

    def start(self, run_name: str, tags: Dict[str, str]):
        self.mlflow.start_run(run_name=run_name, tags=tags)
        self._active = True

    def log_params(self, params: Dict[str, object]):
        self.mlflow.log_params({
            k: ("" if v is None else v)
            for k, v in params.items()
        })

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step: Optional[int] = None):
        for k, v in metrics.items():
            if v is not None:
                self.mlflow.log_metric(k,
                                       float(v),
                                       step=step if step is not None else 0)

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None):
        path = Path(path)
        if path.is_file():
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self,
                      dir_path: Path,
                      artifact_path: Optional[str] = None):
        dir_path = Path(dir_path)
        if dir_path.is_dir():
            self.mlflow.log_artifacts(str(dir_path),
                                      artifact_path=artifact_path)

    def end(self):
        if self._active:
            self.mlflow.end_run()
            self._active = False


class WandbLogger(BaseLogger):

    def __init__(self, cfg):
        super().__init__(cfg)
        import wandb
        self.wandb = wandb
        self.run = None

    def start(self, run_name: str, tags: Dict[str, str]):
        self.run = self.wandb.init(
            project=self.cfg.logger.project,
            name=run_name,
            tags=list(tags.values()),  # wandb tags are a list; keep it simple
            settings=self.wandb.Settings(start_method="thread"),
        )

    def log_params(self, params: Dict[str, object]):
        if self.run: self.run.config.update(params, allow_val_change=True)

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step: Optional[int] = None):
        if self.run:
            self.wandb.log(metrics if step is None else {
                **metrics, "_step": step
            })

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None):
        # Simple: log file into run; for directories use log_artifacts
        if self.run and Path(path).is_file():
            self.wandb.save(str(path))

    def log_artifacts(self,
                      dir_path: Path,
                      artifact_path: Optional[str] = None):
        # Use an Artifact for directories
        if not self.run or not Path(dir_path).is_dir(): return
        art = self.wandb.Artifact(
            name=
            f"{self.cfg.encoder.name}-{self.cfg.prep.tile_size}-{self.cfg.mag_label}",
            type=artifact_path or "artifacts")
        art.add_dir(str(dir_path))
        self.run.log_artifact(art)

    def end(self):
        if self.run:
            self.run.finish()
            self.run = None


def make_logger(cfg) -> BaseLogger:
    try:
        if cfg.logger.backend == "mlflow":
            return MLflowLogger(cfg)
        if cfg.logger.backend == "wandb":
            return WandbLogger(cfg)
    except ImportError:
        # if the package isn't installed, silently fall back to no-op
        pass
    return NullLogger(cfg)
