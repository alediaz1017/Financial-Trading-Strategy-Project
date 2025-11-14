import matplotlib.pyplot as plt


class Visualizer:
    """Handles visualization of training metrics"""
    
    def plot_training_metrics(self, history, model_name="MODEL"):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_losses'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Mean Squared Error', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_title(f'{model_name} - Loss Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # MAPE plot
        ax2.plot(history['train_mapes'], label='Train MAPE', linewidth=2)
        ax2.plot(history['val_mapes'], label='Validation MAPE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAPE (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.set_title(f'{model_name} - MAPE Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()