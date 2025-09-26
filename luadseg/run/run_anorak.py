import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from luadseg.embed.compute import embed_tiles
from luadseg.eval.anorak import run_linear_cv_and_eval
from luadseg.prep.anorak.prepare import run_prep
from luadseg.run.loggers import make_logger
from luadseg.run.utils import should_skip_embed, should_skip_prep


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="anorak",
)
def main(cfg: DictConfig):
    logger = make_logger(cfg)
    run_name = f"{cfg.encoder.name}_{cfg.prep.tile_size}_{cfg.mag_label}"
    tags = {
        "run_group": str(cfg.run_group),
        "encoder": cfg.encoder.name,
        "tile_size": str(cfg.prep.tile_size),
        "mag_label": cfg.mag_label,
        "cv": str(cfg.evaluation.tile_head.cv),
        "seed": str(cfg.run.seed),
        "dataset": cfg.dataset.name,
    }
    logger.start(run_name, tags)
    logger.log_params({
        "epochs":
        cfg.evaluation.tile_head.train.epochs,
        "lr":
        cfg.evaluation.tile_head.train.lr,
        "weight_decay":
        cfg.evaluation.tile_head.train.weight_decay,
        "dropout":
        cfg.evaluation.tile_head.train.dropout,
        "temperature":
        cfg.evaluation.tile_head.train.temperature,
        "batch_size":
        cfg.evaluation.tile_head.train.batch_size,
        "bg_class_weight":
        cfg.evaluation.tile_head.train.bg_class_weight,
    })
    # keep a copy of the resolved config
    Path("config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    logger.log_artifact(Path("config.yaml"))
    try:
        # 1) PREP
        if cfg.stage_switches.prep and not should_skip_prep(cfg):
            # run_prep(cfg.prep | {"out_dir": str(prep_outputs(cfg))})
            run_prep(cfg.prep)

        # 2) EMBED
        if cfg.stage_switches.embed and not should_skip_embed(
                cfg, cfg.encoder.name):
            embed_tiles(
                encoder_name=cfg.encoder.name,
                root_dir=Path(cfg.prep.out_dir),
                out_path=Path(cfg.embeddings.h5_path),
                weights_path=cfg.encoder.weights or None,
                batch_size=cfg.embeddings.batch_size,
                num_workers=cfg.embeddings.num_workers,
                device=cfg.encoder.device,
                use_amp=cfg.embeddings.amp,
                dataset_name=cfg.dataset.name,
                apply_torch_scripting=cfg.encoder.apply_torch_scripting,
            )

        # 3) TRAIN/EVAL
        if cfg.stage_switches.train_eval:
            run_linear_cv_and_eval(
                cfg,
                h5_filepath=cfg.embeddings.h5_path,
                encoder_name=cfg.encoder.name,
            )
            # log end-of-job artifacts
            metrics_path = Path("cv_metrics.json")
            oof_path = Path("oof_preds.parquet")
            heads_dir = Path("heads")
            logger.log_artifact(metrics_path, artifact_path="metrics")
            logger.log_artifact(oof_path, artifact_path="preds")
            logger.log_artifacts(heads_dir, artifact_path="heads")

            # (optional) parse a couple of scalars for dashboards
            if metrics_path.exists():
                data = json.loads(metrics_path.read_text())
                for k, v in data.get("tile_level", {}).get("summary",
                                                           {}).items():
                    logger.log_metrics({f"tile/{k}": float(v)})
                for i, v in enumerate(
                        data.get("tile_level", {}).get("fold_js", [])):
                    logger.log_metrics({"tile/fold_js": float(v)}, step=i)
    finally:
        logger.end()


if __name__ == "__main__":
    main()
