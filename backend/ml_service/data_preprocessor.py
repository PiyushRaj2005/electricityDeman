import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def create_time_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def prepare_features(self, data_dict):
        features = []

        if 'temperature' in data_dict:
            features.append(data_dict['temperature'])
        else:
            features.append(25.0)

        if 'humidity' in data_dict:
            features.append(data_dict['humidity'])
        else:
            features.append(60.0)

        if 'windSpeed' in data_dict:
            features.append(data_dict['windSpeed'])
        else:
            features.append(10.0)

        features.append(1 if data_dict.get('isHoliday', False) else 0)
        features.append(1 if data_dict.get('isWeekend', False) else 0)

        now = datetime.now()
        features.append(now.hour / 24.0)
        features.append(now.day / 31.0)
        features.append(now.month / 12.0)
        features.append(now.weekday() / 7.0)
        features.append(np.sin(2 * np.pi * now.hour / 24))

        return np.array(features, dtype=np.float32)

    def generate_mock_historical_data(self, num_days=30):
        date_range = pd.date_range(end=datetime.now(), periods=num_days * 24, freq='H')

        np.random.seed(42)
        base_demand = 1000
        hourly_pattern = np.array([0.6, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 1.1, 1.0, 0.9, 0.9, 0.95,
                                   1.0, 0.95, 0.9, 0.9, 1.0, 1.2, 1.3, 1.2, 1.1, 1.0, 0.8, 0.7])

        demand = []
        for i, dt in enumerate(date_range):
            hour_factor = hourly_pattern[dt.hour]
            weekend_factor = 0.85 if dt.weekday() >= 5 else 1.0
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * dt.month / 12)
            noise = np.random.normal(0, 0.05)

            daily_demand = base_demand * hour_factor * weekend_factor * seasonal_factor * (1 + noise)
            demand.append(daily_demand)

        df = pd.DataFrame({
            'timestamp': date_range,
            'demand': demand,
            'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 365)) + np.random.normal(0, 2, len(date_range)),
            'humidity': 60 + 15 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 30)) + np.random.normal(0, 5, len(date_range)),
            'wind_speed': 10 + 5 * np.random.randn(len(date_range))
        })

        return df

    def create_sequences(self, data, seq_length=24):
        sequences = []
        targets = []

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def aggregate_to_scale(self, df, scale='hourly'):
        if scale == 'hourly':
            return df
        elif scale == 'daily':
            return df.resample('D', on='timestamp').mean()
        elif scale == 'weekly':
            return df.resample('W', on='timestamp').mean()
        elif scale == 'monthly':
            return df.resample('M', on='timestamp').mean()
        else:
            return df
