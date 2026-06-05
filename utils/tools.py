import os
import pandas as pd
import torch
import torch.nn as nn
import math
import logging

from transformers import (
    TrainerCallback, 
    TrainerState, 
    TrainerControl, 
)

# for logging
def get_logger(logger_name=None, log_level=logging.INFO):
    """
    """
    logging.basicConfig(
        filename=log_file,    
        format="%(message)s", 
        level=log_level
    )
    
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(log_level) 
        
    return logger


def printArgs(args):
    # Print arguments line by line 
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
    
    print("")


def get_train_metrics(log_history):
    num_entries = len(log_history)
    last_row = log_history[num_entries-1]

    best_eval_loss = float("inf")
    epoch_best_loss = 1
    crsp_train_loss = None

    # get best eval loss
    for entry in log_history:
        if "eval_loss" in entry:
            if entry["eval_loss"] < best_eval_loss:
                best_eval_loss = entry["eval_loss"]
                epoch_best_loss = int(entry["epoch"])

    # get train loss corresponding to best eval loss
    for entry in log_history:
        if "loss" in entry and int(entry["epoch"]) == epoch_best_loss:
            crsp_train_loss = entry["loss"]

    return best_eval_loss, crsp_train_loss, epoch_best_loss


def save_results(
        run_id,
        result_path, 
        train_metrics, 
        best_eval_loss, 
        crsp_train_loss, 
        epoch_best_loss, 
        test_results
    ):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name = "stats.txt"
    result_path = os.path.join(result_path, file_name)
    
    with open(result_path, "w") as f:
        print(f"{run_id}\n", file=f)
        print(f"<----- Training stats ----->", file=f)
        print(f"{train_metrics}\n", file=f)
        print(f"<----- Training results ----->", file=f)
        print(f"Best eval loss: {best_eval_loss:.3f}", file=f)
        print(f"Corresponding train loss: {crsp_train_loss}", file=f)
        print(f"Epoch of best eval loss: {epoch_best_loss}", file=f)
        
        # test results
        print("<----- Test result ----->:", file=f)
        print(f"{test_results}", file=f)


class GradNormLoggerCallback(TrainerCallback):
    """A custom callback to log the gradient norm manually."""

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # We only log on the main process to avoid redundant entries in a distributed setup
        if state.is_world_process_zero and state.is_local_main_process:
            model = kwargs.get('model')
            optimizer = kwargs.get('optimizer')
            
            # Use torch.nn.utils.clip_grad_norm_ to get the norm without clipping
            # We set max_norm to infinity to effectively disable clipping
            if optimizer is not None and model is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=float('inf')
                )
                
                # Log the computed gradient norm
                log_dict = {'grad_norm': total_norm.item()}
                control.log_history.append(log_dict)

