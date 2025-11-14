import torch
import pandas as pd
import numpy as np


class Predictor:
    """Handles future price predictions"""
    
    def __init__(self, model, scaler, data_loader, config):
        self.model = model
        self.scaler = scaler
        self.data_loader = data_loader
        self.config = config
    
    def predict_future(self, model_name="MODEL"):
        """Predict future housing prices for all counties"""
        print("=" * 60)
        print(f"GENERATING {model_name} PREDICTIONS")
        print("=" * 60)
        print(f"Input years................. {', '.join(self.config.FUTURE_INPUT_YEARS)}")
        print(f"Forecast horizon............ {self.config.OUTPUT_LENGTH} years")
        print()
        
        predictions = {}
        
        for county in self.data_loader.df_prices_only.index:
            series = self.data_loader.df_prices_only.loc[county][
                self.config.FUTURE_INPUT_YEARS
            ].values.astype(float)
            
            if len(series) < self.config.INPUT_LENGTH:
                continue
            
            # Prepare input
            input_seq = series.reshape(-1, 1)
            input_seq_scaled = self.scaler.transform(input_seq).reshape(
                1, self.config.INPUT_LENGTH, 1
            )
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(
                    torch.tensor(input_seq_scaled, dtype=torch.float32)
                ).numpy()
            
            # Inverse transform
            pred = self.scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).reshape(pred_scaled.shape)
            
            predictions[county] = pred[0].tolist()
        
        df_predictions = pd.DataFrame({
            "County": list(predictions.keys()),
            "Predicted_Prices": list(predictions.values())
        })
        
        print(f"Counties predicted.......... {len(predictions)}")
        print()
        
        return df_predictions
    
    def display_predictions(self, df_predictions, model_name="MODEL"):
        """Display predictions in formatted style"""
        print("=" * 60)
        print(f"{model_name} PREDICTIONS SUMMARY")
        print("=" * 60)
        
        for idx, row in df_predictions.iterrows():
            county = row['County']
            prices = row['Predicted_Prices']
            
            print(f"\nCounty: {county}")
            print("-" * 40)
            
            # Get historical prices for context
            hist_prices = self.data_loader.df_prices_only.loc[county][
                self.config.FUTURE_INPUT_YEARS
            ].values
            
            print("HISTORICAL PRICES:")
            for year, price in zip(self.config.FUTURE_INPUT_YEARS, hist_prices):
                print(f"  {year}..................... ${price:,.2f}")
            
            print("\nPREDICTED PRICES:")
            start_year = int(self.config.FUTURE_INPUT_YEARS[-1]) + 1
            for i, price in enumerate(prices):
                year = start_year + i
                print(f"  {year}..................... ${price:,.2f}")
            
            # Calculate growth
            avg_hist = np.mean(hist_prices)
            avg_pred = np.mean(prices)
            growth = ((avg_pred - avg_hist) / avg_hist) * 100
            
            print(f"\nAVERAGE FORECAST GROWTH..... {growth:+.2f}%")
        
        print("\n" + "=" * 60)
        print()