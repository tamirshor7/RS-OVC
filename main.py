import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict

from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset

# --- Repository Path Injection ---
def _initialize_repo_path(repo_dir="CountGD"):
    """
    Ensures the repository directory is in the system path to allow internal module resolution.
    """
    abs_repo_path = os.path.abspath(repo_dir)
    if not os.path.exists(abs_repo_path):
        print(f"[Warning] Repository folder '{repo_dir}' not found in current directory: {os.getcwd()}")
        return

    if abs_repo_path not in sys.path:
        sys.path.insert(0, abs_repo_path)
        print(f"[Info] Added '{abs_repo_path}' to sys.path")

# Inject path immediately
_initialize_repo_path("CountGD")

# --- Internal Imports ---

import CountGD.util.misc as utils
from CountGD.util.get_param_dicts import get_param_dict
from CountGD.util.logger import setup_logger
from CountGD.util.slconfig import DictAction, SLConfig
from CountGD.util.utils import BestMetricHolder
from CountGD.groundingdino.util.utils import clean_state_dict
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch


def filter_checkpoint_weights(model, state_dict, excluded_prefixes):
    """
    Loads weights into the model while filtering out keys starting with specific prefixes.
    Useful for partial loading or transfer learning when architectures differ slightly.
    """
    filtered_dict = {}
    skipped_keys = 0
    
    for key, val in state_dict.items():
        # Skip if key starts with any of the excluded prefixes
        if any(key.startswith(prefix) for prefix in excluded_prefixes):
            skipped_keys += 1
            logging.warning(f"Skipping parameter loading for: {key}")
        else:
            filtered_dict[key] = val

    if skipped_keys:
        logging.warning(f"Excluded {skipped_keys} parameters based on prefixes: {excluded_prefixes}")

    # Load with strict=False to allow for the missing keys we just filtered
    return model.load_state_dict(filtered_dict, strict=False)


def model_init(args):
    """
    Selects and builds the appropriate model architecture based on the mode argument.
    """
    # Import the specific builder function based on the mode
    if args.mode == "fused":
        from models.GroundingDINO.fused_gdino_dinov3 import build_fused_gdino as model_builder
    else:
        from models.GroundingDINO.gdino_dinov3 import build_groundingdino as model_builder

    model, loss_fn, post_processors = model_builder(args)
    return model, loss_fn, post_processors


def parse_cli_arguments():
    """
    Defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser("RS-OVC Experiment Runner", add_help=False)
    
    # Config and options
    parser.add_argument("--config_file", "-c", type=str, default="/home/tamirshor_google_com/CountGD/config/cfg_fndd.py")
    parser.add_argument("--options", nargs="+", action=DictAction, help="Override config settings (key=value)")
    parser.add_argument("--output_dir", default="out_rsovc", help="Directory for saving results")
    parser.add_argument("--note", default="", help="Experiment notes")
    
    # Dataset settings
    parser.add_argument("--datasets", type=str, default="/home/tamirshor_google_com/CountGD/config/datasets_shared_fndd.json")
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--num_exemplars", default=3, type=int)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # Training modes & Hyperparams
    parser.add_argument("--mode", type=str, default='fused', choices=['rs_only', 'rsft', 'fused','countgd'], 
                        help="Baseline selection: 'countgd', 'rs_only', 'rsft', or 'fused' (RS-OVC)")
    parser.add_argument("--train_with_exemplar_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    parser.add_argument("--pretrain_model_path", default="/home/tamirshor_google_com/RS-OVC/checkpoints/checkpoint_fsc147_best.pth")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--modality_dropout", action="store_true", help="Randomly drop text or visual inputs")

    # Evaluation / Testing flags
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")
    
    # Test-time normalization & cropping
    parser.add_argument("--sam_tt_norm", action="store_true", help="Use SAM for test-time normalization")
    parser.add_argument("--sam_model_path", default="./checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--exemp_tt_norm", action="store_true", help="Use visual exemplars for normalization")
    parser.add_argument("--crop", action="store_true", default=True, help="Adaptive cropping based on exemplars")
    parser.add_argument("--simple_crop", action="store_true", help="Adaptive cropping without exemplars")
    parser.add_argument("--remove_bad_exemplar", action="store_true", default=True, help="Filter inaccurate annotations")

    # Distributed Training
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local-rank", type=int) # Duplicate for compatibility

    return parser


def run_experiment(args):
    """
    Main execution logic for training and evaluation.
    """
    utils.setup_distributed(args)
    
    # --- Configuration Setup ---
    print(f"Loading configuration: {args.config_file}")
    sl_cfg = SLConfig.fromfile(args.config_file)
    
    if args.options:
        sl_cfg.merge_from_dict(args.options)
        
    if args.rank == 0:
        # Save active config
        os.makedirs(args.output_dir, exist_ok=True)
        sl_cfg.dump(os.path.join(args.output_dir, "config_cfg.py"))
        with open(os.path.join(args.output_dir, "config_args_raw.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Sync config to args
    cfg_dictionary = sl_cfg._cfg_dict.to_dict()
    for k, v in cfg_dictionary.items():
        if not hasattr(args, k):
            setattr(args, k, v)
        else:
            raise ValueError(f"Configuration key collision: {k}")

    # Default debug to False if missing
    if not getattr(args, "debug", None):
        args.debug = False

    # --- Logger Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    experiment_logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"),
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )
    
    
    if args.rank == 0:
        full_config_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(full_config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        experiment_logger.info(f"Full config dumped to {full_config_path}")

    # Load dataset metadata
    with open(args.datasets) as f:
        dataset_registry = json.load(f)

    experiment_logger.info(f"Active Arguments: {args}\n")

    # --- Reproducibility & Device ---
    device = torch.device(args.device)
    rng_seed = args.seed + utils.get_rank()
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    # --- Model Initialization ---
    model, criterion, post_processors = model_init(args)
    model.to(device)

    # DDP Handling
    model_unwrapped = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_unwrapped = model.module

    # --- Optimization Setup ---
    trainable_params = get_param_dict(args, model_unwrapped)

    # Frozen layers logic
    if args.freeze_keywords:
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in args.freeze_keywords):
                param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Data Loaders ---
    if not args.eval:
        train_sets_info = dataset_registry["train"]
        if len(train_sets_info) == 1:
            train_dataset = build_dataset("train", args, train_sets_info[0])
        else:
            # Concatenate multiple datasets
            ds_list = [build_dataset("train", args, info) for info in train_sets_info]
            train_dataset = ConcatDataset(ds_list)

    val_dataset = build_dataset("val", args, dataset_registry["val"][0])

    # Samplers
    if args.distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        if not args.eval:
            train_sampler = DistributedSampler(train_dataset)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        if not args.eval:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

    # Loaders construction
    if not args.eval:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        args.batch_size,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    coco_api_val = get_coco_api_from_dataset(val_dataset)

    # --- Weight Loading ---
    
    # 1. Load frozen weights if specified
    if args.frozen_weights:
        ckpt = torch.load(args.frozen_weights, map_location="cpu")
        model_unwrapped.detr.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)

    # 2. Checkpoint Management (Resume vs Pretrain)
    output_path = Path(args.output_dir)
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.pth")
    
    if args.resume and os.path.exists(checkpoint_file):
        args.resume = checkpoint_file

    if args.resume:
        # Resume training
        if args.resume.startswith("https"):
            ckpt = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            
        model_unwrapped.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)

        if not args.eval and "optimizer" in ckpt and "lr_scheduler" in ckpt and "epoch" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["lr_scheduler"])
            args.start_epoch = ckpt["epoch"] + 1

    elif args.pretrain_model_path:
        # Load from pretrain
        ckpt = torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)["model"]
        
        # Filter keys based on ignore list
        ignore_keywords = args.finetune_ignore if args.finetune_ignore else []
        cleaned_dict = utils.clean_state_dict(ckpt)
        
        filtered_state = OrderedDict()
        for k, v in cleaned_dict.items():
            should_keep = True
            for kw in ignore_keywords:
                if kw in k:
                    should_keep = False
                    break
            if should_keep:
                filtered_state[k] = v
        
        # Special handling for baselines
        ignore_prefixes = []
        if args.mode == 'rs_only':
            # RS only baseline: cannot load input projection due to dim mismatch
            ignore_prefixes = ["input_proj", "feature_map_proj"]
        elif args.mode == 'rsft':
            ignore_prefixes = ["input_proj"]
            
        if ignore_prefixes:
            filter_checkpoint_weights(model_unwrapped, filtered_state, ignore_prefixes)
        else:
            model_unwrapped.load_state_dict(filtered_state, strict=False)

    # --- Evaluation Mode ---
    if args.eval:
        os.environ["EVAL_FLAG"] = "TRUE"
        mae_score, test_stats, coco_eval = evaluate(
            model, model_unwrapped, criterion, post_processors,
            val_loader, coco_api_val, device, args.output_dir, args=args
        )
        
        if args.output_dir:
            utils.save_on_master(coco_eval.coco_eval["bbox"].eval, output_path / "eval.pth")
            
        if args.output_dir and utils.is_main_process():
            log_entry = {**{f"test_{k}": v for k, v in test_stats.items()}}
            with (output_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
        return

    # --- Training Loop ---
    print("Beginning Training Routine...")
    t0 = time.time()
    metric_tracker = BestMetricHolder(init_res=100.0, better="small", use_ema=False)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train Step
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=scheduler, args=args,
            logger=(experiment_logger if args.save_log else None),
        )

        if args.output_dir:
            ckpt_paths = [output_path / "checkpoint.pth"]

        if not args.onecyclelr:
            scheduler.step()

        # Checkpointing
        if args.output_dir:
            # Save periodic checkpoints
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                ckpt_paths.append(output_path / f"checkpoint{epoch:04}.pth")
                
            for path in ckpt_paths:
                save_obj = {
                    "model": model_unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                utils.save_on_master(save_obj, path)

        # Validation Step
        val_mae, val_stats, coco_eval = evaluate(
            model, model_unwrapped, criterion, post_processors,
            val_loader, coco_api_val, device, args.output_dir,
            args=args, logger=(experiment_logger if args.save_log else None),
        )

        # Best Model Tracking
        if metric_tracker.update(val_mae, epoch, is_ema=False):
            best_ckpt_path = output_path / "checkpoint_best_regular.pth"
            utils.save_on_master({
                "model": model_unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }, best_ckpt_path)

        # Logging
        log_data = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in val_stats.items()},
        }
        
        try:
            log_data["now_time"] = str(datetime.datetime.now())
        except Exception:
            pass

        epoch_duration = time.time() - epoch_start
        log_data["epoch_time"] = str(datetime.timedelta(seconds=int(epoch_duration)))

        if args.output_dir and utils.is_main_process():
            with (output_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_data) + "\n")

            # Evaluation details dump
            if coco_eval is not None:
                (output_path / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_eval.coco_eval:
                    save_names = ["latest.pth"]
                    if epoch % 50 == 0:
                        save_names.append(f"{epoch:03}.pth")
                    for name in save_names:
                        torch.save(coco_eval.coco_eval["bbox"].eval, output_path / "eval" / name)

    total_duration = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Total training time: {total_duration}")

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser("RS-OVC", parents=[parse_cli_arguments()])
    args = cli_parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    run_experiment(args)
