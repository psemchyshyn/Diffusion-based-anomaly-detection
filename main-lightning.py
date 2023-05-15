from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel


@hydra.main(version_base=None, config_path="denoising_diffusion_pytorch/diffusion_unconditional/configs", config_name="config")
def run(conf: DictConfig) -> None:
    conf_model = conf["model"]
    conf_data = conf["data"]
    conf_training = conf["training"]
    # conf_anomaly = conf["anomaly"]
    print(OmegaConf.to_yaml(conf))

    lit = LitModel(conf_model, conf_training, conf_data)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{conf_training['save_weight_folder']}",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    logger = WandbLogger(project=conf_training["wandb_project_name"], name=conf_training["wandb_run_name"])
    # logger = TensorBoardLogger(save_dir="temp")
    trainer = Trainer(logger=logger,
                      enable_progress_bar=False,
                      accelerator="gpu",
                      devices=[0],
                      max_epochs=conf_training["epochs"],
                      check_val_every_n_epoch=1,
                      accumulate_grad_batches=conf_training["acc_grad_batches"],
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1,
                      default_root_dir=conf_training['save_logs_dir'],
                      )

    trainer.fit(lit)


if __name__ == "__main__":
    run()
