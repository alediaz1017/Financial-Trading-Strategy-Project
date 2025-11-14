import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, config, scaler):
        self.model = model
        self.config = config
        self.scaler = scaler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_mapes': [],
            'val_mapes': []
        }
    
    def train(self, X_train, y_train, X_val, y_val, model_name="MODEL"):
        """Train the model"""
        print("=" * 60)
        print(f"TRAINING {model_name}")
        print("=" * 60)
        print(f"Epochs...................... {self.config.NUM_EPOCHS}")
        print(f"Learning Rate............... {self.config.LEARNING_RATE}")
        print(f"Optimizer................... Adam")
        print()
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            train_mape = self._calculate_mape(outputs, y_train)
            self.history['train_losses'].append(loss.item())
            self.history['train_mapes'].append(train_mape)
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
            
            val_mape = self._calculate_mape(val_outputs, y_val)
            self.history['val_losses'].append(val_loss.item())
            self.history['val_mapes'].append(val_mape)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS} | "
                      f"Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
                      f"Train MAPE: {train_mape:5.2f}% | Val MAPE: {val_mape:5.2f}%")
        print()
    
    def _calculate_mape(self, predictions, targets):
        """Calculate Mean Absolute Percentage Error"""
        pred_np = predictions.detach().numpy()
        target_np = targets.numpy()
        
        pred_orig = self.scaler.inverse_transform(
            pred_np.reshape(-1, 1)
        ).reshape(pred_np.shape)
        
        target_orig = self.scaler.inverse_transform(
            target_np.reshape(-1, 1)
        ).reshape(target_np.shape)
        
        mape = np.mean(np.abs((target_orig - pred_orig) / target_orig)) * 100
        return mape
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = self.criterion(test_outputs, y_test)
        
        test_mape = self._calculate_mape(test_outputs, y_test)
        
        results = {
            'mse': test_loss.item(),
            'mape': test_mape
        }
        
        # Performance summary
        best_train_mape = min(self.history['train_mapes'])
        best_val_mape = min(self.history['val_mapes'])
        
        print(f"Test MSE.................... {results['mse']:.4f}")
        print(f"Test MAPE................... {results['mape']:.2f}%")
        print(f"Best Train MAPE............. {best_train_mape:.2f}%")
        print(f"Best Val MAPE............... {best_val_mape:.2f}%")
        print()
        
        return results