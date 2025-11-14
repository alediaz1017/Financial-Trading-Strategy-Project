# Configuration Class
class Config:
    """Configuration for housing price forecasting"""
    
    def __init__(self):
        # Data paths
        self.HAI_DATA_PATH = "hai_calculator_results.csv"
        self.PRICES_DATA_PATH = "past_median_household_prices.csv"
        
        # Model parameters
        self.INPUT_LENGTH = 3
        self.OUTPUT_LENGTH = 3
        self.HIDDEN_SIZE_LSTM = 64
        self.HIDDEN_SIZE_RNN = 50
        self.DROPOUT_PROB = 0.2
        
        # Training parameters
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 100
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.85
        
        # Data columns
        self.PRICE_COLUMNS = ['2019', '2020', '2021', '2022', '2023', '2024']
        self.FUTURE_INPUT_YEARS = ['2022', '2023', '2024']
        
        # Output paths
        self.LSTM_OUTPUT = "future_housing_prices_lstm.csv"
        self.RNN_OUTPUT = "future_housing_prices_rnn.csv"


# Import other modules
import sys
sys.path.append('.')

from data.data_loader import DataLoader
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from training.trainer import Trainer
from prediction.predictor import Predictor
from utils.visualizer import Visualizer


def run_model(model_type='lstm'):
    """Run the complete pipeline for specified model type"""
    
    # Configuration
    config = Config()
    
    # Load and prepare data
    data_loader = DataLoader(config)
    data_loader.load_data()
    data = data_loader.prepare_data()
    
    # Initialize model
    if model_type.lower() == 'lstm':
        model = LSTMModel(
            input_size=1,
            hidden_size=config.HIDDEN_SIZE_LSTM,
            output_size=config.OUTPUT_LENGTH
        )
        output_file = config.LSTM_OUTPUT
        model_name = "LSTM"
    else:
        model = RNNModel(
            input_size=1,
            hidden_size=config.HIDDEN_SIZE_RNN,
            output_size=config.OUTPUT_LENGTH,
            dropout_prob=config.DROPOUT_PROB
        )
        output_file = config.RNN_OUTPUT
        model_name = "RNN"
    
    # Train model
    trainer = Trainer(model, config, data_loader.scaler)
    trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        model_name=model_name
    )
    
    # Evaluate model
    trainer.evaluate(data['X_test'], data['y_test'])
    
    # Visualize training
    visualizer = Visualizer()
    visualizer.plot_training_metrics(trainer.history, model_name)
    
    # Make predictions
    predictor = Predictor(model, data_loader.scaler, data_loader, config)
    predictions = predictor.predict_future(model_name=model_name)
    
    # Display predictions
    predictor.display_predictions(predictions, model_name=model_name)
    
    # Save results
    predictions.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}\n")
    
    return model, trainer, predictor


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("HOUSING PRICE FORECASTING SYSTEM")
    print("=" * 60)
    print()
    
    # Run LSTM model
    print("Starting LSTM Model Pipeline...")
    print()
    lstm_model, lstm_trainer, lstm_predictor = run_model('lstm')
    
    print("\n" + "=" * 60)
    print()
    
    # Run RNN model
    print("Starting RNN Model Pipeline...")
    print()
    rnn_model, rnn_trainer, rnn_predictor = run_model('rnn')
    
    print("\n" + "=" * 60)
    print("ALL MODELS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()