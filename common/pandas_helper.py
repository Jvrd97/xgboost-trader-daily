import pandas as pd
import numpy as np
import openpyxl as pxl
from path_man import *
import matplotlib.pyplot as plt

class PandasHelper:
    def __init__(self):
        pass
    @staticmethod
    def excel_open(file):
        book = pxl.open(f"{file}", data_only=True)
        sheet = book.active
        data = []

        for row in sheet.iter_rows(values_only=True):
            data.append(row)

        df = pd.DataFrame(data)

        return df
    @staticmethod
    def rename_col(df ,old_name, new_name):
        return df.rename(columns={old_name: new_name})
    
    @staticmethod
    def full_date(df, name_date_col, format):
        df[name_date_col] = pd.to_datetime(df[name_date_col],
                                             format=format)  # always look at date format "%d.%m.%Y"
        start = df[name_date_col].min()
        end = df[name_date_col].max()
        date_range = pd.date_range(start=start, end=end, freq='D')

        full_dates = pd.DataFrame({name_date_col: date_range})

        # Schritt 2: merge mit originalem df, mit „left“ oder „right“ je nachdem
        df_full_dates = df.merge(full_dates, on=name_date_col, how='right')
        return df_full_dates
    
    @staticmethod
    def date_range(df, name_date_col, format):
        df[name_date_col] = pd.to_datetime(df[name_date_col],
                                           format=format)  # always look at date format "%d.%m.%Y"
        start = df[name_date_col].min()
        end = df[name_date_col].max()
        date_range = pd.date_range(start=start, end=end, freq='D')
        return date_range
    
    @staticmethod
    def calculate_peaks(df, col, strong_mult=3, extreme_mult=5):
        """
        Автоматически создаёт пик-индикаторы на основе статистики колонки.

        moderate_mult — во сколько выше 75-го перцентиля считать "нормальный" пик
        strong_mult   — во сколько выше 75-го считать сильный пик
        extreme_mult  — экстремальный пик
        """

        q75 = df[col].quantile(0.75)
        mean = df[col].mean()
        std = df[col].std()

        # Сильный пик: значения, которые намного превышают уровень
        df[f"{col}_peak_strong"] = (df[col] > q75 * strong_mult).astype(int)

        # Экстремальный пик
        df[f"{col}_peak_extreme"] = (df[col] > q75 * extreme_mult).astype(int)
        if df[f"{col}_peak_extreme"].sum() == 0:
            df = df.drop(columns=[f"{col}_peak_extreme"])

        # Можно добавить ещё "аномалия" через Z-оценку
        zscore = (df[col] - mean) / std
        df[f"{col}_zscore_peak"] = (zscore > 3).astype(int)

        return df
    
    @staticmethod
    def plot_everyting(df, col_name_main, second_col, title, label1, label2):
        # 3. Plot

        plt.figure(figsize=(25, 5))
        plt.plot(df[col_name_main], label=label1, linewidth=2)
        if second_col is not None:
            plt.plot(df[second_col], label=label2, linewidth=2)
        else:
            pass
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()
        
    @staticmethod
    def add_lags(df, col_name, lags=14):
        for i in range(1, lags + 1):
            df[f"{col_name}_lag_{i}"] = df[col_name].shift(i)

        return df
    
    @staticmethod
    def chief_proc(y_true, y_pred, col_true='True', col_pred='Pred', verbose=True):
        """
        Calculates the Mean Percentage Error (MPE) with a specific outlier filter.
        Formula: (True - Pred) / True * 100
        Filter: excludes errors worse than -600% (massive over-predictions).
        """
        # 1. Ensure inputs are Pandas Series to align indices
        y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true.copy()
        y_pred = pd.Series(y_pred, index=y_true.index) if not isinstance(y_pred, pd.Series) else y_pred.copy()

        # 2. Create DataFrame
        df_full = pd.DataFrame({
            col_true: y_true,
            col_pred: y_pred
        })

        # 3. Remove rows where Actual is 0 to avoid DivisionByZero errors
        # Using np.isclose is safer for floats than != 0.0
        df_full = df_full[~np.isclose(df_full[col_true], 0.0)]

        # 4. Calculate Percentage Difference
        # Positive result = Underprediction (True > Pred)
        # Negative result = Overprediction (True < Pred)
        df_full["procent"] = (df_full[col_true] - df_full[col_pred]) / df_full[col_true] * 100

        # 5. Apply the custom filter (legacy logic: keep only values > -600%)
        # This filters out cases where the prediction was > 7x the actual value
        #df_filtered = df_full[df_full["procent"] > -600.0]

        # 6. Calculate Mean
        mean_val = df_full["procent"].mean()

        if verbose:
            print(f"Mean Deviation (MPE) over {len(df_full)} days: {mean_val:.4f}")

        return mean_val
