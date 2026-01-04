
from datetime import datetime, timedelta, date
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# --- БАЗОВЫЕ ФУНКЦИИ-КАЛЬКУЛЯТОРЫ (ДВИГАТЕЛЬ) ---

def calculate_rsi_custom(close_prices: pd.Series, length: int = 20) -> pd.Series:
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Используем EWM для более "pandas-like" расчета среднего прироста/потерь
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands_custom(close_prices: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    log_close = np.log1p(close_prices)
    bb_mid = log_close.rolling(window=length).mean()
    bb_std = log_close.rolling(window=length).std()
    bb_high = bb_mid + (bb_std * std)
    bb_low = bb_mid - (bb_std * std)
    return pd.DataFrame({'bb_low': bb_low, 'bb_mid': bb_mid, 'bb_high': bb_high})

def calculate_normalized_atr_custom(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series,
                                    length: int = 14) -> pd.Series:
    prev_close = close_prices.shift(1)
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - prev_close)
    tr3 = abs(low_prices - prev_close)
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.ewm(alpha=1 / length, adjust=False).mean()
    normalized_atr = (atr - atr.mean()) / atr.std()
    return normalized_atr

def calculate_normalized_macd_custom(close_prices: pd.Series, fast: int = 12, slow: int = 26,
                                     signal: int = 9) -> pd.Series:
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    normalized_macd = (macd_line - macd_line.mean()) / macd_line.std()
    return normalized_macd


#groupby(level=1) we use this only when we have multiindex
# here all functions mostly for Ml and we will see shift almost everywhere
class CustomIndicator:

    def __init__(self):
        pass

    # --- ФУНКЦИИ-ТРАНСФОРМЕРЫ ДЛЯ DATAFRAME ---
    # Если нужен сдвиг (для избежания look-ahead bias):
    @staticmethod
    def add_rsi(df: pd.DataFrame, close_col: str = 'close', length: int = 14) -> pd.DataFrame:
        """
        Calculate RSI with proper column naming based on length parameter.
        """
        df = df.copy()
        
        # Calculate RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        
        rs = gain / loss
        rsi_values = 100 - (100 / (1 + rs))
        
        # Create column name based on length: rsi_14, rsi_7, etc.
        col_name = f'rsi_{length}'
        df[col_name] = rsi_values  # Don't shift here - let the caller decide
        
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, close_col: str = 'close', length: int = 20,
                            std: float = 2.0) -> pd.DataFrame:
        """
        Рассчитывает Полосы Боллинджера и добавляет колонки 'bb_low', 'bb_mid', 'bb_high'.
        """
        df = df.copy()
        bbands_df = calculate_bollinger_bands_custom(df[close_col], length=length)

        # Добавляем каждую колонку отдельно со сдвигом
        df['bb_low'] = bbands_df['bb_low'].shift(1)
        df['bb_mid'] = bbands_df['bb_mid'].shift(1)
        df['bb_high'] = bbands_df['bb_high'].shift(1)

        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close',
                length: int = 14) -> pd.DataFrame:
        """
        Рассчитывает нормализованный ATR и добавляет колонку 'atr'.
        """
        df = df.copy()

        atr_values = calculate_normalized_atr_custom(
            high_prices=df[high_col],
            low_prices=df[low_col],
            close_prices=df[close_col],
            length=length
        )

        df['atr'] = atr_values.shift(1)
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, close_col: str = 'close', 
                fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD indicator.
        """
        df = df.copy()
        
        # Calculate MACD
        ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    @staticmethod
    def add_dollar_volume(df: pd.DataFrame, close_col: str = 'adj close', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Рассчитывает денежный объем и добавляет колонку 'dollar_volume'.
        """
        df = df.copy()  # <- явно создаём копию
        volume_values = (df[close_col] * df[volume_col]) / 1_000_000
        df['dollar_volume'] = volume_values.shift(1)
        return df
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, close_col: str = 'close') -> pd.DataFrame:
        """Добавляет индикаторы momentum и тренда"""
        df = df.copy()

        # Rate of Change (ROC)
        df['roc_5'] = df[close_col].pct_change(5).shift(1)
        df['roc_10'] = df[close_col].pct_change(10).shift(1)
        df['roc_20'] = df[close_col].pct_change(20).shift(1)

        # Moving Average convergence

        # Price position relative to recent high/low
        df['high_20'] = df[close_col].rolling(20).max().shift(1)
        df['low_20'] = df[close_col].rolling(20).min().shift(1)
        df['price_position'] = ((df[close_col] - df['low_20']) /
                                (df['high_20'] - df['low_20'])).shift(1)

        return df

class HelperFin:
    def __init__(self, df=None):
        if df is not None:
            self.df = df
        else:
            self.df = None

    def load_ticker_df(self, ticker_name, time, api_key, resample_freq: str = None):
        api_key = api_key
        ticker = ticker_name             # тикер акции
        start_date = (datetime.now() - timedelta(days=time)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        resample_freq = resample_freq

        if resample_freq is not None:

            url = (
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
                f"?startDate={start_date}&endDate={end_date}&token={api_key}&resampleFreq={resample_freq}"
            )
        else:
            url = (
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
                f"?startDate={start_date}&endDate={end_date}&token={api_key}"
            )

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True, drop=True)
            df = df.to_csv(f"{ticker_name}_raw.csv")

            #df = df.set_index('date').sort_index()
            return df

        else:
            print(f"Error: {response.status_code}, {response.text}")

    def plot_everyting(self, df, col_name_main, second_col = None, title=None, label1=None, label2=None):
        # 3. Plot

        plt.figure(figsize=(25, 5))
        plt.plot(df[col_name_main], label=label1, linewidth=2)
        if second_col is not None:
            plt.plot(df[second_col], label=label2, linewidth=2)
        else:
            pass

        if title is not None:
            plt.title(title)
        else:
            pass
        plt.tight_layout()
        plt.xlabel("date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

    def add_lags(self, df, col_name, lags: int = 14):
        for i in range(1, lags + 1):
            df[f"{col_name}_lag_{i}"] = df[col_name].shift(i)
        return df

    def date_range_from_df(self, df, name_date_col, format=None):
        if format is None:
            df[name_date_col] = pd.to_datetime(df[name_date_col], format=format)
        else:
            df[name_date_col] = pd.to_datetime(df[name_date_col])
        # always look at date format "%d.%m.%Y"
        start = df[name_date_col].min()
        end = df[name_date_col].max()
        date_range = pd.date_range(start=start, end=end, freq='D')
        return date_range

    def simple_date_range(self, start, end, freq='D'):
        start = date(start)
        end = date(end)
        date_range = pd.date_range(start=start, end=end, freq=freq)
        return date_range

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    def calculate_correlations_with_lags(
            df: pd.DataFrame,
            target_col: str,
            predictor_col: str,
            lags: list = [30, 60, 90, 120],
            use_log_returns: bool = True
    ):
        """
        Рассчитывает корреляции между двумя временными рядами с разными лагами.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с данными (должен быть отсортирован по времени)
        target_col : str
            Название колонки целевой переменной (например, 'close_WLD')
        predictor_col : str
            Название колонки предиктора (например, 'close_BTC')
        lags : list
            Список лагов в днях для сдвига предиктора
        use_log_returns : bool
            Если True - использовать логдоходности, иначе - сырые цены

        Returns:
        --------
        dict : словарь с результатами корреляций для каждого лага
        """

        df_work = df.copy()

        # Если используем логдоходности
        if use_log_returns:
            df_work[f'{target_col}_logret'] = np.log(df_work[target_col] / df_work[target_col].shift(1))
            df_work[f'{predictor_col}_logret'] = np.log(df_work[predictor_col] / df_work[predictor_col].shift(1))
            target = f'{target_col}_logret'
            predictor_base = f'{predictor_col}_logret'
        else:
            target = target_col
            predictor_base = predictor_col

        results = {}

        # Корреляция без лага (lag=0)
        results['lag_0'] = df_work[[target, predictor_base]].corr().iloc[0, 1]

        # Корреляции с лагами
        for lag in lags:
            predictor_lagged = f'{predictor_base}_lag{lag}'
            df_work[predictor_lagged] = df_work[predictor_base].shift(lag)

            # Pearson correlation
            corr = df_work[[target, predictor_lagged]].corr().iloc[0, 1]
            results[f'lag_{lag}'] = corr

        return results, df_work

    def plot_correlation_heatmap(results_dict: dict, title: str = "Correlation with Lags"):
        """
        Визуализация корреляций в виде тепловой карты.

        Parameters:
        -----------
        results_dict : dict
            Словарь с результатами корреляций {lag: correlation_value}
        title : str
            Заголовок графика
        """
        lags = list(results_dict.keys())
        correlations = list(results_dict.values())

        plt.figure(figsize=(12, 4))

        # Barplot
        plt.subplot(1, 2, 1)
        plt.bar(lags, correlations, color='steelblue', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Lag (days)')
        plt.ylabel('Pearson Correlation')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # Heatmap
        plt.subplot(1, 2, 2)
        corr_matrix = np.array(correlations).reshape(1, -1)
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    xticklabels=lags, yticklabels=['Correlation'],
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Heatmap')

        plt.tight_layout()
        plt.show()

    def compare_price_vs_logreturns_correlation(
            df: pd.DataFrame,
            target_col: str,
            predictor_col: str,
            lags: list = [30, 60, 90, 120]
    ):
        """
        Сравнивает корреляции для сырых цен и логдоходностей.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с данными
        target_col : str
            Колонка таргета (WLD)
        predictor_col : str
            Колонка предиктора (BTC)
        lags : list
            Список лагов

        Returns:
        --------
        pd.DataFrame : таблица сравнения корреляций
        """

        # Корреляции для сырых цен
        results_price, _ = calculate_correlations_with_lags(
            df, target_col, predictor_col, lags, use_log_returns=False
        )

        # Корреляции для логдоходностей
        results_logret, _ = calculate_correlations_with_lags(
            df, target_col, predictor_col, lags, use_log_returns=True
        )

        # Создаём сравнительную таблицу
        comparison = pd.DataFrame({
            'Lag': list(results_price.keys()),
            'Correlation_Price': list(results_price.values()),
            'Correlation_LogReturns': list(results_logret.values())
        })

        # Визуализация
        plt.figure(figsize=(12, 5))

        x = np.arange(len(comparison))
        width = 0.35

        plt.bar(x - width / 2, comparison['Correlation_Price'], width,
                label='Raw Prices', alpha=0.7, color='steelblue')
        plt.bar(x + width / 2, comparison['Correlation_LogReturns'], width,
                label='Log Returns', alpha=0.7, color='darkorange')

        plt.xlabel('Lag')
        plt.ylabel('Pearson Correlation')
        plt.title(f'{target_col} vs {predictor_col}: Price vs LogReturns Correlation')
        plt.xticks(x, comparison['Lag'], rotation=45)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        return comparison


class GeneralHelper:
    def __init__(self):
        pass

    def file_path(self, path):
        if os.path.exists(path):
            print(f"✓ Path exists: {path}")
            print(f"· Absolute path: {os.path.abspath(path)}")

            if os.path.isdir(path):
                print("· Type: Directory")
            elif os.path.isfile(path):
                print("· Type: File")
            else:
                print("· Type: Other (symbolic link, device, etc.)")

            return True

        else:
            print(f"✗ Path does NOT exist: {path}")
            print(f"· Parent directory: {os.path.dirname(path)}")
            return False