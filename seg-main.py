import os.path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from segmentation.discriminative_lightning import SegLit
import hydra


@hydra.main(version_base=None, config_path="segmentation/configs", config_name="config")
def run(conf: DictConfig) -> None:
    conf_data = conf["data"]
    conf_anomaly = conf["anomaly"]
    conf_training = conf["training"]

    print(OmegaConf.to_yaml(conf))

    already_trained = False
    if os.path.exists(conf_training['save_weight_folder']):
        already_trained = True
        lit = SegLit.load_from_checkpoint(os.path.join(conf_training['save_weight_folder'], "last.ckpt"), conf_training=conf_training, conf_data=conf_data, conf_anomaly=conf_anomaly).cuda()
    else:
        lit = SegLit(conf_training, conf_data, conf_anomaly)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=conf_training['save_weight_folder'],
        filename="model-{epoch:02d}",
        save_top_k=1,
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
    if not already_trained:
        trainer.fit(lit)
    trainer.test(lit)


if __name__ == "__main__":
    run()
