"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""

import os
import time
import yaml
import shutil
import operator
import itertools
from os.path import join
from loguru import logger
import numpy as np
from functools import reduce
from yacs.config import CfgNode as CN
from typing import List, Union
from flatten_dict import flatten, unflatten

# Please update your own paths to dataset folders here:
H36M_ROOT = "data/human36m"
RICH_ROOT = "data/RICH/"
LSP_ROOT = "data/LSP"
LSP_ORIGINAL_ROOT = "data/LSP_ORIGINAL"
LSPET_ROOT = "data/hr-lspet"
MPII_ROOT = "data/MPII-pose"
COCO_ROOT = "data/COCO/images"
MPI_INF_3DHP_ROOT = "data/mpi_inf_3dhp"
PW3D_ROOT = "data/3DPW___v1"
UPI_S1H_ROOT = ""
AGORA_ROOT = ""

# Output folder to save test/train npz files
DATASET_NPZ_PATH = "data/dataset_extras"

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate
# the .npz files with the annotations.
OPENPOSE_PATH = ""

# Path to test/train npz files
DATASET_FILES = [
    {
        "h36m-p1": join(DATASET_NPZ_PATH, "h36m_valid_protocol1.npz"),
        "h36m-p2": join(DATASET_NPZ_PATH, "h36m_valid_protocol2.npz"),
        "h36m-test-s1": join(DATASET_NPZ_PATH, "h36m_only_S1_valid_protocol1.npz"),
        "rich-val": join(DATASET_NPZ_PATH, "rich_world_val_onlycam0_new.npz"),
        "rich-test": join(DATASET_NPZ_PATH, "rich_world_test_new.npz"),
        "rich-test-onlycam0": join(
            DATASET_NPZ_PATH, "rich_world_test_onlycam0_new.npz"
        ),
        "rich-test-last2seq-onlycam0": join(
            DATASET_NPZ_PATH, "rich_world_test_last2seq_only_cam0_new.npz"
        ),
        "lsp": join(DATASET_NPZ_PATH, "lsp_dataset_test.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_valid.npz"),
        "3dpw": join(DATASET_NPZ_PATH, "3dpw_test.npz"),
        "agora_val": join(DATASET_NPZ_PATH, "agora_val_0.npz"),
    },
    {
        # 'h36m': join(DATASET_NPZ_PATH, 'h36m_train_world.npz'),
        "rich": join(DATASET_NPZ_PATH, "rich_world_val_new.npz"),
        # 'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),
        "agora1": join(DATASET_NPZ_PATH, "agora_train_0.npz"),
        "agora2": join(DATASET_NPZ_PATH, "agora_train_1.npz"),
        "agora3": join(DATASET_NPZ_PATH, "agora_train_2.npz"),
        "agora4": join(DATASET_NPZ_PATH, "agora_train_3.npz"),
        "agora5": join(DATASET_NPZ_PATH, "agora_train_4.npz"),
        "lsp-orig": join(DATASET_NPZ_PATH, "lsp_dataset_original_train.npz"),
        "mpii": join(DATASET_NPZ_PATH, "mpii_train.npz"),
        "coco": join(DATASET_NPZ_PATH, "coco_2014_train.npz"),
        "lspet": join(DATASET_NPZ_PATH, "hr-lspet_train.npz"),
        "mpii-eft": join(DATASET_NPZ_PATH, "mpii_train_eft.npz"),
        "coco-eft": join(DATASET_NPZ_PATH, "coco_2014_train_eft.npz"),
        "lspet-eft": join(DATASET_NPZ_PATH, "hr-lspet_train_eft.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_train.npz"),
    },
]

DATASET_FOLDERS = {
    "h36m": H36M_ROOT,
    "rich": RICH_ROOT,
    # 'rich-val-onlycam0': RICH_ROOT,
    # 'rich-test-onlycam0': RICH_ROOT,
    "rich-val": RICH_ROOT,
    "rich-test": RICH_ROOT,
    "rich-test-onlycam0": RICH_ROOT,
    "rich-val-onlycam1": RICH_ROOT,
    "rich-val-onlycam5": RICH_ROOT,
    "rich-test-last2seq-onlycam0": RICH_ROOT,
    # 'agora': AGORA_ROOT,
    "agora1": AGORA_ROOT,
    "agora2": AGORA_ROOT,
    "agora3": AGORA_ROOT,
    "agora4": AGORA_ROOT,
    "agora5": AGORA_ROOT,
    "agora_val": AGORA_ROOT,
    "h36m-p1": H36M_ROOT,
    "h36m-p2": H36M_ROOT,
    "h36m-test-s1": H36M_ROOT,
    "h36m-train-small": H36M_ROOT,
    "lsp-orig": LSP_ORIGINAL_ROOT,
    "lsp": LSP_ROOT,
    "lspet": LSPET_ROOT,
    "lspet-eft": LSPET_ROOT,
    "mpi-inf-3dhp": MPI_INF_3DHP_ROOT,
    "mpii": MPII_ROOT,
    "mpii-eft": MPII_ROOT,
    "coco": COCO_ROOT,
    "coco-eft": COCO_ROOT,
    "3dpw": PW3D_ROOT,
    "upi-s1h": UPI_S1H_ROOT,
}

CUBE_PARTS_FILE = "data/cube_parts.npy"
JOINT_REGRESSOR_TRAIN_EXTRA = "data/J_regressor_extra.npy"
JOINT_REGRESSOR_H36M = "data/J_regressor_h36m.npy"
VERTEX_TEXTURE_FILE = "data/vertex_texture.npy"
STATIC_FITS_DIR = "data/static_fits"
SMPL_MEAN_PARAMS = "data/smpl_mean_params.npz"
SMPL_MODEL_DIR = "data/smpl"

##### CONFIGS #####
hparams = CN()

hparams.name = "default"
hparams.time_to_run = np.inf
hparams.resume = False
hparams.num_workers = 8
hparams.pin_memory = True
hparams.log_dir = "logs"
hparams.checkpoint = None
hparams.from_json = None
hparams.pretrained_checkpoint = None
hparams.num_epochs = 200
hparams.lr = 5e-5
hparams.cop_w = 10.0
hparams.cop_k = 100.0
hparams.in_alpha1 = 1.0
hparams.in_alpha2 = 0.5
hparams.out_alpha1 = 1.0
hparams.out_alpha2 = 0.15
hparams.contact_thresh = 0.1
hparams.batch_size = 64
hparams.summary_steps = 100
hparams.test_steps = 1000
hparams.run_eval_h36m = False
hparams.run_eval_rich_val = False
hparams.run_eval_rich_test = False
hparams.run_eval_3dpw = False
hparams.checkpoint_steps = 10000
hparams.img_res = 224
hparams.rot_factor = 30
hparams.noise_factor = 0.4
hparams.scale_factor = 0.25
hparams.ignore_3d = False
hparams.is_agora = False
hparams.is_others = False
hparams.shape_loss_weight = 0
hparams.keypoint_loss_weight = 5.0
hparams.pose_loss_weight = 1.0
hparams.beta_loss_weight = 0.001
hparams.stability_loss_weight = 1.0
hparams.inside_push_loss_weight = 1.0
hparams.outside_pull_loss_weight = 1.0
hparams.openpose_train_weight = 0.0
hparams.gt_train_weight = 1.0
hparams.run_smplify = False
hparams.smplify_threshold = 100.0
hparams.num_smplify_iters = 100
hparams.shuffle_train = True


def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer="path")
    hyper_params = []

    for k, v in flattened_config_dict.items():
        if isinstance(v, list):
            if k in excluded_keys:
                flattened_config_dict[k] = ["+".join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool):
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v, list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split("+")
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ["True", "False"]):
                exp[param_name] = [True if x == "True" else False for x in param_value]
            if param_value in ["True", "False"]:
                if param_value == "True":
                    exp[param_name] = True
                else:
                    exp[param_name] = False

        experiments[exp_id] = unflatten(exp, splitter="path")

    return experiments, hyper_params


def run_grid_search_experiments(
    cfg_id,
    cfg_file,
    use_cluster,
    bid,
    memory,
    script="main.py",
    cmd_opts=[],
    gpu_min_mem=10000,
    gpu_arch=("tesla", "quadro", "rtx"),
):
    cfg = yaml.safe_load(open(cfg_file))
    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    logger.info(f"Grid search hparams: \n {hyperparams}")

    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    logger.info(
        f"======> Number of experiment configurations is {len(different_configs)}"
    )

    config_to_run = CN(different_configs[cfg_id])

    if use_cluster:
        execute_task_on_cluster(
            script=script,
            output_dir=config_to_run.log_dir,
            exp_name=config_to_run.name,
            num_exp=len(different_configs),
            cfg_file=cfg_file,
            bid_amount=bid,
            num_workers=config_to_run.num_workers,
            memory=memory,
            exp_opts=cmd_opts,
            gpu_min_mem=gpu_min_mem,
            gpu_arch=gpu_arch,
        )
        exit()

    # ==== create logdir using hyperparam settings
    logtime = time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = f"{logtime}_{config_to_run.name}"

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split("/"))
        logdir += f'_{hp.replace("/", ".").replace("_", "").lower()}-{v}'

    logdir = os.path.join(config_to_run.log_dir, logdir)

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(logdir, "config.yaml"))

    # add extra paramters for checkpoints and tensorboard
    config_to_run["checkpoint_dir"] = os.path.join(logdir, "checkpoints")
    os.makedirs(config_to_run["checkpoint_dir"], exist_ok=True)
    config_to_run["summary_dir"] = os.path.join(logdir, "tensorboard")
    os.makedirs(config_to_run["summary_dir"], exist_ok=True)

    def save_dict_to_yaml(obj, filename, mode="w"):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)), os.path.join(logdir, "config_to_run.yaml")
    )

    return config_to_run
