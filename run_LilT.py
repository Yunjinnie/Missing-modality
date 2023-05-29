import os
import copy
import warnings
#import torch
import wandb
from torchmetrics.functional import f1_score
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from LilT.lilt_config import ex
from LilT.models.lilt_module import Lilt
from LilT.datamodules.multitask_datamodule import MTDataModule


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    print("data module :",dm)
    model = Lilt(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=False,
        # monitor="val/the_metric",
        monitor='train_loss',
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["text_encoder"]}_text_{_config["vision_encoder"]}_vision',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    #     from pytorch_lightning.profiler import SimpleProfiler
    #     profiler = SimpleProfiler()

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print("batch, per_gpu_batchsize, num_gpus, num_nodes", _config["batch_size"], _config["per_gpu_batchsize"],
          num_gpus, _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1

    trainer = pl.Trainer(
        #gpus=_config["num_gpus"],
        accelerator="gpu", #ddp
        strategy = "ddp",
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        #accelerator="ddp",
        #benchmark=True,
        #deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=WandbLogger(name=_config["wandb_name"], project=_config["wandb_project"]),#logger,
        #prepare_data_per_node=False,
        #replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        #flush_logs_every_n_steps=10,
        #resume_from_checkpoint=_config["resume_from"],
        #weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        #         profiler=profiler,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
