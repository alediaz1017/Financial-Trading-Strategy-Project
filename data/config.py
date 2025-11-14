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