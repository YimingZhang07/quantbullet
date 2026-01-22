import shutil
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quantbullet.torch.gate import BoundedWindowGate

DEV_MODE = False

class TestBoundedWindowGate(unittest.TestCase):

    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        if DEV_MODE:
            # In DEV_MODE, ensure directory exists but don't clear it
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            # In non-DEV_MODE, clear and recreate
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        # only clear cache dir in non-dev mode
        # we want to keep files for inspection in dev mode
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)


    def test_fully_set( self ):
        x: np.ndarray = np.linspace(-20, 20, 100)
        y: np.ndarray = 1 / ( 1 + np.exp(-x) )

        gate = BoundedWindowGate( a=-20, b=20, lower=0, upper=1, s=5 )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate.to(device)
        x_tensor = torch.from_numpy(x).to(device)
        y_tensor = gate(x_tensor)
        y_pred = y_tensor.cpu().numpy()

        # plot the results together
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Actual')
        ax.plot(x, y_pred, label='Predicted')
        ax.legend()
        fig.savefig(Path(self.cache_dir) / "test_bounded_window_gate_fully_set.png")
        plt.close(fig)

    def test_fully_trainable( self ):
        x: np.ndarray = np.linspace(-20, 20, 100)
        y: np.ndarray = 1 / ( 1 + np.exp(-x) )

        gate = BoundedWindowGate()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate.to(device)
        
        x_tensor = torch.from_numpy(x).float().to(device)  # Convert to float32
        y_tensor = torch.from_numpy(y).float().to(device)  # Convert to float32
        
        optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)
        
        print(f"Initial params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        for i in range(200):
            optimizer.zero_grad()
            y_pred = gate(x_tensor)
            loss = torch.mean((y_pred - y_tensor) ** 2)
            loss.backward()
            
            if i == 0:
                # Check gradients on first iteration
                print(f"Gradients: center={gate.center_raw.grad}, lower={gate.lower_raw.grad}, upper={gate.upper_raw.grad}, s={gate.log_s.grad}")
            
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Iteration {i}, loss: {loss.item():.6f}, center={gate.center().item():.3f}, "
                      f"width={gate.width().item():.3f}, s={gate.s().item():.3f}")

        print(f"Final params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        y_pred = gate(x_tensor).detach().cpu().numpy()

        # plot the results together
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Actual')
        ax.plot(x, y_pred, label='Predicted')
        ax.legend()
        ax.set_title(f'Adam Optimizer - Final Loss: {loss.item():.6f}')
        fig.savefig(Path(self.cache_dir) / "test_bounded_window_gate_fully_trainable.png")
        plt.close(fig)

    def test_fully_trainable_lbfgs( self ):
        """Test with LBFGS optimizer - usually faster but can be tricky"""
        x: np.ndarray = np.linspace(-20, 20, 100)
        y: np.ndarray = 1 / ( 1 + np.exp(-x) )

        gate = BoundedWindowGate()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate.to(device)
        
        x_tensor = torch.from_numpy(x).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        
        # LBFGS often works better with smaller learning rate
        optimizer = torch.optim.LBFGS(gate.parameters(), lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
        
        print(f"Initial params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        for i in range(50):
            def closure():
                optimizer.zero_grad()
                y_pred = gate(x_tensor)
                loss = torch.mean((y_pred - y_tensor) ** 2)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            if i % 5 == 0:
                print(f"Iteration {i}, loss: {loss.item():.6f}, center={gate.center().item():.3f}, "
                      f"width={gate.width().item():.3f}, s={gate.s().item():.3f}")

        print(f"Final params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        y_pred = gate(x_tensor).detach().cpu().numpy()

        # plot the results together
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Actual')
        ax.plot(x, y_pred, label='Predicted')
        ax.legend()
        ax.set_title(f'LBFGS Optimizer - Final Loss: {loss.item():.6f}')
        fig.savefig(Path(self.cache_dir) / "test_bounded_window_gate_fully_trainable_lbfgs.png")
        plt.close(fig)

    def test_partial_trainable( self ):
        """Test fitting a portion of x² with a sigmoid - using masked loss for region of interest"""
        # Full range for visualization
        x: np.ndarray = np.linspace(0, 10, 200)
        y: np.ndarray = x ** 0.5

        # Define region of interest: [0, 5] where we want good fit
        roi_mask = (x >= 0) & (x <= 5)

        # Let ALL params be trainable to find best fit for the ROI
        gate = BoundedWindowGate( a=0, b=5 )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate.to(device)
        
        print(f"Target: Fit x² ONLY on [0, 5] (region of interest)")
        print(f"Initial params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        x_tensor = torch.from_numpy(x).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        
        # Use LBFGS for faster convergence
        optimizer = torch.optim.LBFGS(gate.parameters(), lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
        
        for i in range(50):
            def closure():
                optimizer.zero_grad()
                # AUTO MASK: only penalize errors in [a, b] region
                loss = gate.compute_loss(x_tensor, y_tensor, auto_mask=True)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            if i % 5 == 0:
                print(f"Iteration {i}, loss (ROI only): {loss.item():.6f}, center={gate.center().item():.3f}, "
                      f"width={gate.width().item():.3f}, s={gate.s().item():.3f}, "
                      f"lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        # Final predictions
        y_pred = gate(x_tensor).detach().cpu().numpy()

        print(f"Final params: center={gate.center().item():.3f}, width={gate.width().item():.3f}, "
              f"s={gate.s().item():.3f}, lower={gate.lower().item():.3f}, upper={gate.upper().item():.3f}")
        
        # plot the results together with residuals
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Highlight the ROI
        ax1.axvspan(0, 5, alpha=0.2, color='green', label='ROI [0,5]')
        ax1.plot(x, y, label='Target: x²', linewidth=2, color='blue')
        ax1.plot(x, y_pred, label='Sigmoid Fit', linewidth=2, linestyle='--', color='orange')
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True, alpha=0.3)
        
        # Plot residuals
        residuals = y_pred - y
        ax2.axvspan(0, 5, alpha=0.2, color='green', label='ROI [0,5]')
        ax2.plot(x, residuals, color='red', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Residual (Predicted - Actual)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig(Path(self.cache_dir) / "test_bounded_window_gate_partial_trainable.png")
        plt.close(fig)