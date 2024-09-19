import argparse
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

from audiossl.models.atst.audio_transformer import AST_base, AST_small
from audiossl.methods.atst.downstream.model import PretrainedEncoderPLModule
from desed_task.nnet.fusion.fusion_model import ATSTModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet, WeakSet
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.version.sed_fusion_rct import SEDTask4_2021
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

import warnings
warnings.filterwarnings("ignore")


def get_atst_model(net_confs, device):
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

    if net_confs["rct_atst_weights"]:
        print("Covering ATST with: {}".format(net_confs["rct_atst_weights"]))
        rct_weights = torch.load(net_confs["rct_atst_weights"], map_location=device)
        encoder_weights = rct_weights["sed_student"]
        state_dict = {k.replace("encoder.encoder.", ""): v for k, v in encoder_weights.items()}
        atst_encoder.load_state_dict(state_dict, strict=False)


    # Warp by pretrain PL module
    atst_encoder = PretrainedEncoderPLModule(
        pretrained_encoder=atst_encoder,
        chunk_len=net_confs["audio_len"],
        n_blocks=net_confs["n_last_blocks"])

    atst_encoder.freeze()
    if net_confs["unfreeze"] == "all":
        print("Training all ATST layers.")
        atst_encoder.unfreeze()
    elif type(net_confs["unfreeze"]) == int:
        unfreeze_id = list(range(12))[::-1]
        unfreeze_id = unfreeze_id[:net_confs["unfreeze"]]
        print("Unfreezing:", unfreeze_id)
        for i in unfreeze_id:
            layer = getattr(atst_encoder.encoder.blocks, str(i))
            for child in layer.parameters():
                child.requires_grad = True
    else:
        print("Freezing the pretraining model.")
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

def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    print("Test only", test_only, "Evaluation:", evaluation)
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    # elif test_only:
    #     dsets = ["test_folder"]
    else:
        dsets = ["eval_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def single_run(
        config,
        log_dir,
        gpus,
        strong_real=True,
        checkpoint_resume=None,
        test_state_dict=None,
        fast_dev_run=False,
        evaluation=False
):
    """
    Running sound event detection baselin

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
    print("Using strong real data:", strong_real)
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
    sed_student, pretraining = get_atst_model(config["net"], device=torch.device("cuda:{}".format(gpus[0])))

    pp = 0
    for p in list(sed_student.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print("Total parameters:", pp)

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        if strong_real:
            strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
            strong_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                strong_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )

        if strong_real:
            strong_full_set = torch.utils.data.ConcatDataset([strong_set, synth_set])
            tot_train_data = [strong_full_set, weak_set, unlabeled_set]
        else:
            tot_train_data = [synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset([synth_val, weak_val])

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                        config["training"]["batch_size"][indx]
                        * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        opt = torch.optim.Adam(sed_student.parameters(), 1e-3, betas=(0.9, 0.999))
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
        )
        print(f"experiment dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="val/obj_metric",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4_2021(
        config,
        encoder=encoder,
        sed_student=sed_student,
        device=gpus,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
        chunk_input=config["net"]["chunk_input"]
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=gpus,
        strategy="ddp_find_unused_parameters_false",
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    if test_state_dict is None:
        trainer.fit(desed_training)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
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
        default="0,",
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
        "--eval_from_checkpoint",
        default=None,
        help="Evaluate the model specified"
    )
    parser.add_argument(
        "--strong_real",
        action="store_true",
        default=False,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)
    print(configs)
    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint, map_location=torch.device("cuda:{}".format(args.gpus[0])))
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

    test_only = test_from_checkpoint is not None
    resample_data_generate_durations(configs["data"], test_only, evaluation)
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.strong_real,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation
    )
