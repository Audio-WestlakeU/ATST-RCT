import argparse
import sys
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from audiossl.models.atst.audio_transformer import AST_base, AST_small
from audiossl.methods.atst.downstream.model import PretrainedEncoderPLModule
from desed_task.nnet.fusion.fusion_model import ATSTModel

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet, WeakSet
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.ensemble import EnsembleModule
from local.version.sed_fusion_rct import SEDTask4_2021
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

import warnings
warnings.filterwarnings("ignore")


def get_atst_model(net_confs, device=None):
    # Sanity check
    assert net_confs["atst_mode"] in ["small", "base"], \
        "Please select ATST mode from 'small' or 'base'; Other versions are still under-developed."
    # Load ATST model
    atst_encoder = AST_small(**net_confs["ast"]) if net_confs["atst_mode"] == "small" else AST_base(**net_confs["ast"])

    # If using pretrained model
    pretraining = False
    if net_confs["pretrained_ckpt_path"]:
        from audiossl.methods.atst.downstream.utils import load_pretrained_weights
        # pretrained_encoder = torch.load(net_confs["pretrained_ckpt_path"])  # Still under consideration [MARK]
        load_pretrained_weights(
            atst_encoder,
            pretrained_weights=net_confs["pretrained_ckpt_path"],
            checkpoint_key="teacher",
            device=device)
        pretraining = True

    # Warp by pretrain PL module
    atst_encoder = PretrainedEncoderPLModule(
        pretrained_encoder=atst_encoder,
        chunk_len=net_confs["audio_len"],
        n_blocks=net_confs["n_last_blocks"])
    print("Freezing the ATST model.")
    atst_encoder.freeze()

    atst_model = ATSTModel(encoder=atst_encoder,
                           atst_mode=net_confs["atst_mode"],
                           dim=atst_encoder.embed_dim,
                           chunk_input=net_confs["chunk_input"],
                           **net_confs["crnn"])

    if net_confs["pretrained_crnn"]:
        print("Loading pretrained CNN model from {}".format(net_confs["pretrained_crnn"]))
        crnn_state_dict = torch.load(net_confs["pretrained_crnn"], map_location="cpu")
        # Left CNN parameter only
        cnn_state_dict = {k: v for k, v in crnn_state_dict['sed_student'].items() if "cnn" in k}
        atst_state_dict = atst_model.state_dict()
        atst_state_dict.update(cnn_state_dict)
        atst_model.load_state_dict(atst_state_dict, strict=False)
        if net_confs["freeze_cnn"]:
            print("Freezing the CNN model.")
            for child in atst_model.cnn.parameters():
                child.requires_grad = False
        else:
            print("Unfreezing the CNN model.")
    return atst_model, pretraining


def resample_data_generate_durations(config_data, test_only=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    else:
        dsets = ["test_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    for base_set in ["synth_val", "test"]:
        if not os.path.exists(config_data[base_set + "_dur"]) or computed:
            generate_tsv_wav_durations(
                config_data[base_set + "_folder"], config_data[base_set + "_dur"]
            )


def single_run(
    config,
    log_dir,
    gpus,
    paths,
    evaluation,
    checkpoint_resume=None,
    fast_dev_run=False,
):
    """
    Running sound event detection baseline

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"]
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder,
            pad_to=None,
            return_filename=True
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    base_model, _ = get_atst_model(config["net"], device=torch.device("cuda:{}".format(gpus[0])))
    # config["net"]["atst_mode"] = "small"
    # config["net"]["pretrained_ckpt_path"] = "./pretraining/small_checkpoint0300.pth"
    # small_model, _ = get_atst_model(config["net"])

    train_dataset = None
    valid_dataset = None
    batch_sampler = None
    opt = None
    exp_scheduler = None
    logger = True
    callbacks = None

    desed_training = EnsembleModule(
        paths=paths,
        base_model=base_model,
        small_model=None,
        cls=SEDTask4_2021,
        hparams=config,
        encoder=encoder,
        evaluation=evaluation,
        opt=opt,
        device=gpus,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run
    ).to(torch.device("cuda:{}".format(gpus[0])))

    flush_logs_every_n_steps = 100
    log_every_n_steps = 40
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    n_epochs = config["training"]["n_epochs"]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        gpus=gpus,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    # desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/sed.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2021_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
        "so uses one GPU indexed by 0.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)
    # print(configs)
    test_from_checkpoint = args.test_from_checkpoint
    test_model_state_dict = None
    paths = configs["augs"]["ensemble"]
    if not paths:
        sys.exit("Please input the ensemble models")

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    test_only = True
    resample_data_generate_durations(configs["data"], test_only)
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        paths,
        args.eval,
        args.resume_from_checkpoint,
        test_model_state_dict,
    )
