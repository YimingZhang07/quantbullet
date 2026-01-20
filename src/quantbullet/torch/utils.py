import torch
import torch.nn as nn
from typing import Callable

def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)

def train_model_lbfgs(
    model: nn.Module,
    loss_fn: nn.Module,
    forward_fn: Callable,
    steps: int = 100,
    lr: float = 1.0,
    max_iter: int = 25,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    log_every: int = 10,
    verbose: bool = True,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
) -> list:
    """
    Generic LBFGS training loop.
    
    Args:
        model: PyTorch model to train
        loss_fn: Loss function (e.g., nn.HuberLoss(), EpsilonInsensitiveLoss())
        forward_fn: Function that takes model and returns (predictions, targets)
                    Signature: forward_fn(model) -> (yhat, y)
        steps: Number of LBFGS steps
        lr: Learning rate (typically 1.0 for LBFGS)
        max_iter: Max iterations per LBFGS step
        tolerance_grad: Gradient tolerance
        tolerance_change: Parameter change tolerance
        log_every: Log loss every N steps
        verbose: Whether to print progress
        early_stopping: Stop if loss doesn't improve for `patience` steps
        patience: Number of steps to wait before stopping
        min_delta: Minimum loss improvement to reset patience
        
    Returns:
        losses: List of loss values at each step
        
    Example:
        # Define how to compute predictions
        def forward_fn(model):
            yhat = model(x_mvoc, bucket_idx, wap=x_wap, cpnspread=x_cpn)
            return yhat, y_target
        
        losses = train_model_lbfgs(
            model=model,
            loss_fn=loss_fn,
            forward_fn=forward_fn,
            steps=100,
            lr=1.0,
        )
    """
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )
    
    losses = []
    best_loss = float("inf")
    no_improve_steps = 0
    
    for step in range(1, steps + 1):
        model.train()
        
        def closure():
            opt.zero_grad(set_to_none=True)
            yhat, y = forward_fn(model)
            loss = loss_fn(yhat, y)
            loss.backward()
            return loss
        
        loss = opt.step(closure)
        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val + min_delta < best_loss:
            best_loss = loss_val
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if verbose and step % log_every == 0:
            print(f"[LBFGS] Step {step:04d}/{steps} | Loss: {loss_val:.6f}")

        if early_stopping and no_improve_steps >= patience:
            if verbose:
                print(
                    f"[LBFGS] Early stopping at step {step:04d} | "
                    f"Best loss: {best_loss:.6f}"
                )
            break
    
    return losses