from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from classification.discriminative_lightning import ClassifierLit
import hydra


@hydra.main(version_base=None, config_path="classification/configs", config_name="config")
def run(conf: DictConfig) -> None:
    conf_data = conf["data"]
    conf_anomaly = conf["anomaly"]
    conf_training = conf["training"]

    print(OmegaConf.to_yaml(conf))

    lit = ClassifierLit(conf_training, conf_data, conf_anomaly)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{conf_training['save_weight_folder']}",
        filename="model-{epoch:02d}",
        save_top_k=5,
        save_last=True,
        mode="min",
    )

    logger = WandbLogger(project=conf_training["wandb_project_name"], name=conf_training["wandb_run_name"])


    trainer = Trainer(logger=logger,
                      accelerator="gpu",
                      devices=[0],
                      max_epochs=conf_training["epochs"],
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1,
                      default_root_dir=conf_training['save_logs_dir'],
                      )

    trainer.fit(lit)


if __name__ == "__main__":
    run()
