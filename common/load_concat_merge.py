import os
import pandas as pd
from dotenv import load_dotenv
from common.path_man import *

load_dotenv()

def concat_single_ticker(ticker: str, interval: str) -> tuple[pd.DataFrame, str | None]:
    """
    1. Загружает основную историю (Concat файл или Cutoff файл при первом запуске).
    2. Проверяет TMP папку на наличие нового файла.
    3. Если есть новый файл -> Конкатенирует -> Перезаписывает Concat файл.
    4. Возвращает ПОЛНЫЙ обновленный DataFrame.
    """
    
    # --- 1. ПУТИ ---
    # Папка, куда падают новые файлы (из MoveOneFile)
    data_tmp_folder = BASE_TMP_PATH / interval /ticker
    
    # Папка, где лежит накопленная история
    concat_folder =  BASE_CONCAT_PATH / interval / ticker
    
    # Файл, который мы будем перезаписывать (Накопительный)
    concat_file_path = os.path.join(concat_folder, f"{ticker}_{interval}_concat.csv")
    
    # Файл "затравка" (Исходная история, если concat еще не создан)
    seed_file_path = os.path.join(concat_folder, f"{ticker}_{interval}_concat.csv")

    file_to_delete = None
    df_history = pd.DataFrame()
    df_new = pd.DataFrame()

    # --- 2. ЗАГРУЗКА ИСТОРИИ (BASE) ---
    if os.path.exists(concat_file_path):
        print(f"  [{ticker}] Loading existing CONCAT DB...")
        df_history = pd.read_csv(concat_file_path)
    elif os.path.exists(seed_file_path):
        print(f"  [{ticker}] First run? Loading SEED history ({os.path.basename(seed_file_path)})...")
        df_history = pd.read_csv(seed_file_path)
    else:
        print(f"✗ [{ticker}] CRITICAL: No History found! (Checked concat and seed)")
        return pd.DataFrame(), None

    # Парсинг дат истории
    if not df_history.empty:
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])

    # --- 3. ЗАГРУЗКА НОВОГО ФАЙЛА (TMP) ---
    new_files = []
    if data_tmp_folder and os.path.exists(data_tmp_folder):
        new_files = [f for f in os.listdir(data_tmp_folder) if f.endswith('.csv')]

    if new_files:
        # Берем первый попавшийся (или сортируем по имени)
        matching_file = sorted(new_files)[0]
        full_tmp_path = os.path.join(data_tmp_folder, matching_file)
        
        print(f"→ [{ticker}] Found NEW data in TMP: {matching_file}")
        
        df_new = pd.read_csv(full_tmp_path)
        
        # Очистка мусора
        if 'Unnamed: 0' in df_new.columns: 
            df_new.drop(columns=['Unnamed: 0'], inplace=True)
            
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        
        # Помечаем файл на удаление
        file_to_delete = full_tmp_path
    else:
        print(f"  [{ticker}] No new data in TMP. Using existing history.")

    # --- 4. КОНКАТЕНАЦИЯ И ПЕРЕЗАПИСЬ ---
    if not df_new.empty:
        # Объединяем
        df_total = pd.concat([df_history, df_new], ignore_index=True)
        
        # Удаляем дубликаты (оставляем последние, если вдруг файл пришел дважды)
        df_total = df_total.drop_duplicates(subset=['timestamp'], keep='last')
        df_total = df_total.sort_values('timestamp').reset_index(drop=True)
        
        # Очистка колонок перед сохранением
        df_total = df_total.loc[:, ~df_total.columns.str.contains('^Unnamed')]

        # ПЕРЕЗАПИСЫВАЕМ ФАЙЛ
        os.makedirs(concat_folder, exist_ok=True)
        df_total.to_csv(concat_file_path, index=False)
        
        print(f"  [{ticker}] UPDATED & OVERWRITTEN: {concat_file_path}")
        print(f"  [{ticker}] Total rows: {len(df_total)} (Added {len(df_new)})")
        
        return df_total, file_to_delete
    else:
        # Если новых данных нет, просто возвращаем историю
        return df_history, None


def merge_tickers(df_main: pd.DataFrame, df_secondary: pd.DataFrame, 
                  main_ticker: str, secondary_ticker: str,
                  lag_days: int = 90) -> pd.DataFrame:
    """
    Мерджит полные истории двух тикеров с учетом лага.
    """
    print(f"\n→ Merging {main_ticker} + {secondary_ticker} (lag={lag_days} days)")
    
    if df_main.empty or df_secondary.empty:
        print("✗ Merge failed: One of the DataFrames is empty.")
        return pd.DataFrame()

    df_main = df_main.copy()
    df_secondary = df_secondary.copy()
    
    # Переименовываем колонки второго тикера
    rename_map = {col: f"{col}_{secondary_ticker}" for col in df_secondary.columns if col != 'timestamp'}
    df_secondary = df_secondary.rename(columns=rename_map)
    
    # Применяем Лаг к BTC (сдвигаем будущее в прошлое, чтобы сопоставить)
    # Пример: Чтобы предсказать WLD сегодня, используя BTC 90 дней назад.
    # Мы прибавляем 90 дней к дате BTC. Теперь старая цена BTC имеет сегодняшнюю дату.
    df_secondary['timestamp'] = df_secondary['timestamp'] + pd.Timedelta(days=lag_days)
    
    # Inner Merge (пересечение)
    df_merged = pd.merge(df_main, df_secondary, on='timestamp', how='inner')
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    # Убираем Unnamed если вылезли
    df_merged = df_merged.loc[:, ~df_merged.columns.str.contains('^Unnamed')]
    
    print(f"  Merged size: {len(df_merged)} rows")
    if not df_merged.empty:
        print(f"  Range: {df_merged['timestamp'].min().date()} to {df_merged['timestamp'].max().date()}")
    
    return df_merged


def load_concat_merge(main_ticker: str, secondary_ticker: str, 
                      interval: str, lag_days: int = 90) -> tuple[pd.DataFrame, list]:
    
    # Папка для сохранения итогового merge файла
    MERGE_FOLDER = MERGE_DATA / interval / f"{main_ticker}_{secondary_ticker}"
    os.makedirs(MERGE_FOLDER, exist_ok=True)
    # Имя файла, который мы будем ПЕРЕЗАПИСЫВАТЬ
    MERGE_FILE = os.path.join(MERGE_FOLDER, f"{main_ticker}_{secondary_ticker}_{interval}_merged.csv")
    
    files_to_delete = []
    
    print(f"\n{'=' * 50}")
    print(f"PIPELINE: LOAD -> CONCAT (Overwrite) -> MERGE (Overwrite)")
    print(f"{'=' * 50}")
    
    # 1. Обработка Main Ticker (WLD) - возвращает ПОЛНУЮ обновленную историю
    df_main, file_main = concat_single_ticker(main_ticker, interval)
    if file_main: files_to_delete.append(file_main)
    
    # 2. Обработка Secondary Ticker (BTC) - возвращает ПОЛНУЮ обновленную историю
    df_secondary, file_secondary = concat_single_ticker(secondary_ticker, interval)
    if file_secondary: files_to_delete.append(file_secondary)
    
    # 3. Мердж
    df_merged = merge_tickers(df_main, df_secondary, main_ticker, secondary_ticker, lag_days)
    
    # 4. Перезапись Merge файла
    if not df_merged.empty:
        os.makedirs(MERGE_FOLDER, exist_ok=True)
        df_merged.to_csv(MERGE_FILE, index=False)
        print(f"\n✓ MERGED FILE OVERWRITTEN: {MERGE_FILE}")
    else:
        print("\n✗ Merge resulted in empty DataFrame (Check data overlap or lags).")

    return df_merged, files_to_delete


def delete_temp_files(files_to_delete: list):
    """Безопасное удаление файлов из TMP."""
    if not files_to_delete: 
        print("\n(No temp files to delete)")
        return
        
    print("\nCleaning up temp files...")
    for filepath in files_to_delete:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"✓ Deleted: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"⚠ Could not delete {filepath}: {e}")