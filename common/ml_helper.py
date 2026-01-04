import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from datetime import datetime
import os
import json
import glob
import json


class MLHelper:
    def __init__(self, y_true, y_pred):
        # сохраняем только Series одной длины
        self.y_true = pd.Series(y_true).reset_index(drop=True)
        self.y_pred = pd.Series(y_pred).reset_index(drop=True)

    # --------- Метрики ---------
    def rmse(self):
        return root_mean_squared_error(self.y_true, self.y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def mape(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def metrics(self):
        return {
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "MAPE (%)": self.mape()
        }

    # --------- График ---------
    def plot(self, title="Model Evaluation"):
        plt.figure(figsize=(25, 5))
        plt.plot(self.y_true, label="True", linewidth=2)
        plt.plot(self.y_pred, label="Predicted", linewidth=2)
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

    # --------- Всё вместе ---------
    def evaluate(self, title="Model Evaluation"):
        self.plot(title)
        return self.metrics()
    
    @staticmethod
    def simple_forecast_metrics(y_true, y_pred):
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
        print(f"RMSE: {rmse}")
        
        # 2. FIX THE PLOTTING INDEX
        # Check if y_true is a Pandas Object to get the correct dates/index
        if hasattr(y_true, 'index'):
            plot_index = y_true.index
            y_true_values = y_true.values
        else:
            plot_index = range(len(y_true))
            y_true_values = y_true

        plt.figure(figsize=(25, 5))
        
        # Pass 'plot_index' as the X-axis for BOTH lines
        plt.plot(plot_index, y_true_values, label="True", linewidth=2)
        plt.plot(plot_index, y_pred, label="Predicted", linewidth=2, linestyle='--')
        
        plt.title("Model Evaluation")
        plt.xlabel("Index / Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        

        # MAPE (Mean Absolute Percentage Error) - средняя абсолютная процентная ошибка
        # Осторожно: не работает когда y_true близко к 0
        
    # evaluate model
    @staticmethod
    def evaluate_forecast_full(predictions, y_true, df_real_future, forecaster, features, target,  to_json=True):
    
            # === 1. ИЗВЛЕЧЕНИЕ ДАННЫХ ===
        # Convert predictions to list/array
        if isinstance(predictions, dict):
            predictions = list(predictions.values())
        elif isinstance(predictions, (pd.Series, pd.DataFrame)):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)
        
        # Convert y_true to array
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        elif isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        elif isinstance(y_true, list):
            y_true = np.array(y_true)  # ✅ FIX: lists to numpy array
        else:
            y_true = np.array(y_true)
        
        # ✅ CRITICAL: Ensure df_real_future has same length as predictions
        if len(df_real_future) != len(predictions):
            print(f"⚠️  WARNING: df_real_future has {len(df_real_future)} rows, "
                f"but predictions has {len(predictions)} values.")
            print(f"   Using only first {len(predictions)} rows of df_real_future")
            df_real_future = df_real_future.iloc[:len(predictions)]

        # === 2. РАСЧЁТ МЕТРИК ОШИБОК ===

        # MAE (Mean Absolute Error) - средняя абсолютная ошибка
        mae = mean_absolute_error(y_true, predictions)

        # RMSE (Root Mean Squared Error) - корень из средней квадратичной ошибки
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
            
        r2 = r2_score(y_true, predictions)

        # MAPE (Mean Absolute Percentage Error) - средняя абсолютная процентная ошибка
        # Осторожно: не работает когда y_true близко к 0
        mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100

        # Direction Accuracy - точность предсказания направления (вверх/вниз)
        direction_pred = np.sign(predictions)  # +1 если рост, -1 если падение
        direction_true = np.sign(y_true)
        direction_accuracy = np.mean(direction_pred == direction_true) * 100

        # Средняя ошибка (может быть положительной или отрицательной)
        mean_error = np.mean(predictions - y_true)

        # === 3. ВЫВОД РЕЗУЛЬТАТОВ ===
        print("=" * 60)
        print("МЕТРИКИ КАЧЕСТВА ПРОГНОЗА")
        print("=" * 60)
        print(f"MAE (средняя абсолютная ошибка):     {mae:.6f}")
        print(f"R2:                                  {r2:.6f}")
        print(f"RMSE (корень из средней кв. ошибки): {rmse:.6f}")
        print(f"MAPE (ошибка в %):                   {mape:.2f}%")
        print(f"Mean Error (систематическая ошибка): {mean_error:.6f}")
        print(f"Direction Accuracy (точность знака): {direction_accuracy:.2f}%")
        print("=" * 60)
         # assert need to be made
         
        # === 4. ДЕТАЛЬНАЯ ТАБЛИЦА ПО ДНЯМ ===
        comparison_df = pd.DataFrame({
            'Дата': df_real_future.index,
            'Прогноз': predictions,
            'Реальность': y_true,
            'Ошибка': predictions - y_true,
            'Абс. Ошибка': np.abs(predictions - y_true),
            'Ошибка %': np.abs((y_true - predictions) / (y_true + 1e-8)) * 100,
            'Направление OK': direction_pred == direction_true
        })

        print("\nДЕТАЛЬНАЯ ТАБЛИЦА:")
        print(comparison_df.to_string(index=False))

        # === 5. ВИЗУАЛИЗАЦИЯ ОШИБОК ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # График 1: Прогноз vs Реальность
        axes[0, 0].plot(y_true, label='Реальность', marker='o', color='green')
        axes[0, 0].plot(predictions, label='Прогноз', marker='x', color='red', linestyle='--')
        axes[0, 0].set_title('Прогноз vs Реальность')
        axes[0, 0].set_xlabel('День')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # График 2: Scatter plot (идеальная линия = диагональ)
        axes[0, 1].scatter(y_true, predictions, alpha=0.7)
        axes[0, 1].plot([y_true.min(), y_true.max()],
                        [y_true.min(), y_true.max()],
                        'r--', label='Идеальный прогноз')
        axes[0, 1].set_xlabel('Реальные Returns')
        axes[0, 1].set_ylabel('Прогнозные Returns')
        axes[0, 1].set_title('Scatter Plot: Прогноз vs Реальность')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # График 3: Ошибки по дням
        axes[1, 0].bar(range(len(y_true)), predictions - y_true,
                    color=['red' if e > 0 else 'green' for e in (predictions - y_true)])
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('Ошибка по дням (Прогноз - Реальность)')
        axes[1, 0].set_xlabel('День')
        axes[1, 0].set_ylabel('Ошибка')
        axes[1, 0].grid(True)

        # График 4: Абсолютные ошибки
        axes[1, 1].bar(range(len(y_true)), np.abs(predictions - y_true), color='orange')
        axes[1, 1].set_title('Абсолютная ошибка по дням')
        axes[1, 1].set_xlabel('День')
        axes[1, 1].set_ylabel('|Ошибка|')
        axes[1, 1].axhline(y=mae, color='red', linestyle='--', label=f'MAE = {mae:.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

        # === 6. СВОДНАЯ СТАТИСТИКА ===
        print("\n" + "=" * 60)
        print("СВОДНАЯ СТАТИСТИКА:")
        print("=" * 60)
        print(f"Правильно предсказано направление: {np.sum(direction_pred == direction_true)} из {len(y_true)} дней")
        print(f"Максимальная ошибка:               {np.max(np.abs(predictions - y_true)):.6f}")
        print(f"Минимальная ошибка:                {np.min(np.abs(predictions - y_true)):.6f}")
        print(f"Стандартное отклонение ошибки:     {np.std(predictions - y_true):.6f}") 
        
        
    def save_evaluation_to_json(predictions,
                                y_true,
                                df_real_future,
                                forecaster,
                                FEATURES,
                                target,
                                mae,
                                mape,
                                rmse):
            # === 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В JSON ===
        # 1. Вспомогательная функция для очистки параметров (чтобы JSON не ругался на numpy типы)
        def safe_serialize(val):
            if isinstance(val, (np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.float64, np.float32)):
                return float(val)
            if isinstance(val, (str, int, float, bool, type(None))):
                return val
            return str(val) # Для всего остального (функции, классы) возвращаем строку

        # 2. Сбор параметров с каждой под-модели (по шагам)
        # forecaster.models - это словарь {step: pipeline}
        model_params_by_step = {}
        try:
            for step, model_step in forecaster.models.items():
                # Берем параметры победившей модели на этом шаге
                raw_params = model_step.get_params()
                # Очищаем их для JSON
                clean_params = {k: safe_serialize(v) for k, v in raw_params.items()}
                model_params_by_step[str(step)] = clean_params
        except AttributeError:
            print("⚠️ Внимание: forecaster.models не найден. Возможно, модель не была обучена.")
            model_params_by_step = "Not trained or invalid object"


        # 3. Формирование итогового словаря
        results = {
            'date': df_real_future.index[0].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            # Метрики (предполагается, что переменные mae, rmse и т.д. уже рассчитаны ранее)
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'mean_error': float(mean_error),
            'direction_accuracy': float(direction_accuracy),

            # Прогнозы
            'predicted': predictions.tolist(),
            'actual': y_true.tolist(),
            'errors': (predictions - y_true).tolist(),
            'directions_correct': (direction_pred == direction_true).tolist(),

            # Features и Model INFO
            'features_used': FEATURES,
            'num_features': len(FEATURES),

            # ! ИЗМЕНЕНИЯ ЗДЕСЬ !
            'model_type': type(forecaster).__name__, # Имя вашего класса (AutoTunedDirectForecaster)
            'base_model_type': type(forecaster.base_pipeline.steps[-1][1]).__name__, # Имя внутренней модели (например HistGradientBoostingRegressor)

            # Сохраняем параметры по шагам: "1": {...}, "2": {...}
            'model_params_per_step': model_params_by_step,
            # Можно добавить общие настройки грида, если они сохранены в классе
            'grid_search_config': {
                'horizon': forecaster.horizon,
                'cv_splits': forecaster.cv_splits
            },

            # Дополнительная инфо
            'target': target,
            'forecast_horizon': FORECAST_HORIZON,

            # Статистика
            'summary': {
                'correct_directions': int(np.sum(direction_pred == direction_true)),
                'total_days': int(len(y_true)),
                'max_error': float(np.max(np.abs(y_pred - y_true))),
                'min_error': float(np.min(np.abs(y_pred - y_true))),
                'std_error': float(np.std(y_pred - y_true))
            }
        }

        # Сохраняем
        filename = f"data_tests/MODEL800/result_{df_real_future.index[0].strftime('%Y%m%d')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Результаты сохранены в {filename}")
            
    @staticmethod
    def analyze_all_predictions(folder=None, results:list = None, target:str = "VM"):
        """
        Полный анализ всех прогнозов из папки data_tests
        """

        # === 1. ЗАГРУЗКА ВСЕХ РЕЗУЛЬТАТОВ ===
        all_results = []
        json_files = glob.glob(f'{folder}/result_*.json')

        if not json_files:
            print(f"❌ Нет файлов в папке {folder}/")
            return None

        print(f"📁 Найдено файлов: {len(json_files)}")

        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_results.append(data)
            except Exception as e:
                print(f"⚠️ Ошибка при чтении {file}: {e}")

        # Сортируем по дате
        all_results = sorted(all_results, key=lambda x: x['date'])

        # === 2. ОБЩАЯ СТАТИСТИКА ===
        total_predictions = len(all_results)
        total_days = sum(r['summary']['total_days'] for r in all_results)
        correct_directions = sum(r['summary']['correct_directions'] for r in all_results)
        overall_accuracy = (correct_directions / total_days) * 100 if total_days > 0 else 0

        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])
        avg_mape = np.mean([
            r['mape'] if r['mape'] <= 1000 else 0
            for r in all_results
        ])

        avg_mean_error = np.mean([r['mean_error'] for r in all_results])

        std_mae = np.std([r['mae'] for r in all_results])
        std_rmse = np.std([r['rmse'] for r in all_results])

        min_mae = min(r['mae'] for r in all_results)
        max_mae = max(r['mae'] for r in all_results)

        # === 3. ВЫВОД ОБЩЕЙ СТАТИСТИКИ ===
        print("\n" + "=" * 80)
        print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ПРОГНОЗАМ")
        print("=" * 80)
        print(f"📊 Период: {all_results[0]['date']} → {all_results[-1]['date']}")
        print(f"📈 Количество прогнозов: {total_predictions}")
        print(f"📅 Всего спрогнозировано дней: {total_days}")
        print(f"\n🎯 ТОЧНОСТЬ НАПРАВЛЕНИЯ:")
        print(f"   ├─ Правильных: {correct_directions} из {total_days}")
        print(f"   └─ Overall Direction Accuracy: {overall_accuracy:.2f}%")
        print(f"\n📉 ОШИБКИ:")
        print(f"   ├─ Средняя MAE:  {avg_mae:.6f} ± {std_mae:.6f}")
        print(f"   ├─ Средняя RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
        print(f"   ├─ Средняя MAPE: {avg_mape:.2f}%")
        print(f"   ├─ Min MAE:      {min_mae:.6f}")
        print(f"   ├─ Max MAE:      {max_mae:.6f}")
        print(f"   └─ Систем. ошибка: {avg_mean_error:.6f}")

        # === 4. СТАТИСТИКА ПО МОДЕЛЯМ ===
        unique_models = set(r.get('model_type', 'Unknown') for r in all_results)
        unique_feature_counts = set(r.get('num_features', 0) for r in all_results)

        print(f"\n🔧 КОНФИГУРАЦИЯ:")
        print(f"   ├─ Модели: {', '.join(unique_models)}")
        print(f"   ├─ Количество features: {', '.join(map(str, sorted(unique_feature_counts)))}")
        print(f"   └─ Target: {all_results[0].get(target, 'Unknown')}")
        print("=" * 80)

        # === 5. ДЕТАЛЬНАЯ ТАБЛИЦА ПО ДНЯМ ===
        print("\n" + "=" * 100)
        print("ДЕТАЛЬНАЯ ИСТОРИЯ ПО ДНЯМ")
        print("=" * 100)
        print(f"{'Дата':<12} {'Dir Acc':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Mean Err':<12} {'Features':<10}")
        print("-" * 100)

        for r in all_results:
            acc = r['direction_accuracy']
            status = '✅' if acc >= 50 else '❌'
            date = r['date']
            mae_val = r['mae']
            rmse_val = r['rmse']
            mape_val = 0 if r['mape'] > 1000 else r['mape']
            mean_err = r['mean_error']
            num_feat = r.get('num_features', 'N/A')

            print(f"{date:<12} {status} {acc:>5.1f}%  {mae_val:>10.6f}  {rmse_val:>10.6f}  {mape_val:>7.2f}%  {mean_err:>10.6f}  {num_feat:>8}")

        print("=" * 100)

        # === 6. СОЗДАЁМ DATAFRAME ===
        df_results = pd.DataFrame([{
            'date': r['date'],
            'direction_accuracy': r['direction_accuracy'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'mape': r['mape'],
            'mean_error': r['mean_error'],
            'num_features': r.get('num_features', 0),
            'model_type': r.get('model_type', 'Unknown')
        } for r in all_results])

        df_results['date'] = pd.to_datetime(df_results['date'])

        # === 7. ВИЗУАЛИЗАЦИЯ ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # График 1: Direction Accuracy по дням
        ax1 = axes[0, 0]
        colors = ['green' if x >= 50 else 'red' for x in df_results['direction_accuracy']]
        ax1.bar(range(len(df_results)), df_results['direction_accuracy'], color=colors, alpha=0.7)
        ax1.axhline(y=50, color='blue', linestyle='--', label='50% (Random)')
        ax1.axhline(y=overall_accuracy, color='orange', linestyle='--', label=f'Average: {overall_accuracy:.1f}%')
        ax1.set_xlabel('Прогноз #')
        ax1.set_ylabel('Direction Accuracy (%)')
        ax1.set_title('Direction Accuracy по дням')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot actual values (combines history + forecast period)
        all_dates = df_results['date'].tolist()
        all_real_values = df_results['real_values'].tolist()

        # Historical line (blue)
        ax.plot(all_dates, all_real_values, 
                linewidth=2, color='steelblue', 
                label='История (Факт)', zorder=1)

        # Forecast points (red with dashed line)
        forecast_dates = all_dates[-len(df_results):]  # Last N days
        forecast_predictions = df_results['predictions'].tolist()

        ax.plot(forecast_dates, forecast_predictions,
                marker='o', linestyle='--', color='red', 
                linewidth=2, markersize=6, alpha=0.8,
                label='Прогноз (AI)', zorder=2)

        # Actual values during forecast (green dots)
        ax.plot(forecast_dates, all_real_values[-len(df_results):],
                marker='o', linestyle='', color='green', 
                markersize=6, alpha=0.8,
                label='Факт после прогноза', zorder=3)

        ax.set_title(f'Walk-Forward Validation: {len(df_results)} итераций', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Returns (VM)')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # График 2: MAE по дням
        ax2 = axes[0, 1]
        ax2.plot(range(len(df_results)), df_results['mae'], marker='o', color='purple', label='MAE')
        ax2.axhline(y=avg_mae, color='red', linestyle='--', label=f'Average: {avg_mae:.6f}')
        ax2.fill_between(range(len(df_results)),
                        avg_mae - std_mae, avg_mae + std_mae,
                        alpha=0.2, color='red', label='±1 STD')
        ax2.set_xlabel('Прогноз #')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE по дням')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Систематическая ошибка (bias)
        ax3 = axes[1, 0]
        ax3.bar(range(len(df_results)), df_results['mean_error'],
                color=['red' if x > 0 else 'green' for x in df_results['mean_error']], alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=avg_mean_error, color='blue', linestyle='--', label=f'Average: {avg_mean_error:.6f}')
        ax3.set_xlabel('Прогноз #')
        ax3.set_ylabel('Mean Error (Bias)')
        ax3.set_title('Систематическая ошибка (Прогноз - Реальность)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # График 4: Cumulative Direction Accuracy
        ax4 = axes[1, 1]
        cumulative_correct = np.cumsum([r['summary']['correct_directions'] for r in all_results])
        cumulative_total = np.cumsum([r['summary']['total_days'] for r in all_results])
        cumulative_accuracy = (cumulative_correct / cumulative_total) * 100
        ax4.plot(range(len(df_results)), cumulative_accuracy, marker='o', color='darkgreen', linewidth=2)
        ax4.axhline(y=50, color='blue', linestyle='--', label='50% (Random)')
        ax4.fill_between(range(len(df_results)), 50, cumulative_accuracy,
                        where=(cumulative_accuracy >= 50), alpha=0.3, color='green', label='Above random')
        ax4.fill_between(range(len(df_results)), 50, cumulative_accuracy,
                        where=(cumulative_accuracy < 50), alpha=0.3, color='red', label='Below random')
        ax4.set_xlabel('Прогноз #')
        ax4.set_ylabel('Cumulative Direction Accuracy (%)')
        ax4.set_title('Накопительная точность направления')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{folder}/analysis_summary.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n✅ График сохранён: {folder}/analysis_summary.png")

        # === 8. СОХРАНЕНИЕ В CSV ===
        df_results.to_csv(f'{folder}/analysis_summary.csv', index=False, encoding='utf-8')
        print(f"✅ Таблица сохранена: {folder}/analysis_summary.csv")

        # === 9. СОХРАНЕНИЕ ОТЧЁТА В MD ===
        report = f"""# Отчёт по прогнозам

    **Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Период:** {all_results[0]['date']} → {all_results[-1]['date']}

    ## Общая статистика

    | Метрика | Значение |
    |---------|----------|
    | Количество прогнозов | {total_predictions} |
    | Всего дней | {total_days} |
    | Правильных направлений | {correct_directions} / {total_days} |
    | **Overall Direction Accuracy** | **{overall_accuracy:.2f}%** |
    | Средняя MAE | {avg_mae:.6f} ± {std_mae:.6f} |
    | Средняя RMSE | {avg_rmse:.6f} ± {std_rmse:.6f} |
    | Средняя MAPE | {avg_mape:.2f}% |
    | Систематическая ошибка | {avg_mean_error:.6f} |

    ## Детальная история

    | Дата | Direction Acc | MAE | RMSE | MAPE | Mean Error | Features |
    |------|---------------|-----|------|------|------------|----------|
    """

        for r in all_results:
            acc = r['direction_accuracy']
            status = '✅' if acc >= 50 else '❌'
            report += f"| {r['date']} | {status} {acc:.1f}% | {r['mae']:.6f} | {r['rmse']:.6f} | {r['mape']:.2f}% | {r['mean_error']:.6f} | {r.get('num_features', 'N/A')} |\n"

        report += f"\n![Analysis Summary]({folder}/analysis_summary.png)\n"

        with open(f'{folder}/REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ Отчёт сохранён: {folder}/REPORT.md")

        return df_results, all_results
    
    
    
    @staticmethod
    def analyze_all_predictions_new(folder=None, results: list = None, target: str = "VM"):
        """
        Полный анализ всех прогнозов
        
        Args:
            folder: Path to folder with result_*.json files (old format)
            results: List of result dictionaries (new format from walk-forward)
        """
        
        all_results = []
        
        # ✅ NEW: If results provided directly, use them
        if results is not None:
            print(f"📊 Analyzing {len(results)} walk-forward results...")
            
            # Convert our format to expected format
            for r in results:
                # Extract summary stats from daily_results
                daily_results = r.get('daily_results', [])
                total_days = len(daily_results)
                correct_directions = sum(1 for d in daily_results if d.get('direction_correct', False))
                
                # Convert to expected format
                converted = {
                    'date': r.get('forecast_start', r.get('train_end_date', 'Unknown')),
                    'direction_accuracy': r.get('direction_accuracy', 0),
                    'mae': r.get('mae', 0),
                    'rmse': r.get('rmse', 0),
                    'mape': r.get('mape', 0),
                    'mean_error': r.get('mean_error', 0),
                    'num_features': r.get('num_features', 0),
                    'model_type': 'Walk-Forward',
                    'predictions': r.get('predictions', []),
                    'real_values': r.get('real_values', []),
                    
                    'summary': {
                        'total_days': total_days,
                        'correct_directions': correct_directions
                    },
                    'target': 'VM'  # You can make this configurable
                }
                all_results.append(converted)
        
        # OLD: Read from JSON files in folder
        elif folder is not None:
            json_files = glob.glob(f'{folder}/result_*.json')
            
            if not json_files:
                print(f"❌ Нет файлов в папке {folder}/")
                return None
            
            print(f"📁 Найдено файлов: {len(json_files)}")
            
            for file in json_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_results.append(data)
                except Exception as e:
                    print(f"⚠️ Ошибка при чтении {file}: {e}")
        
        else:
            print("❌ Необходимо указать либо folder, либо results")
            return None
        
        if not all_results:
            print("❌ Нет данных для анализа")
            return None
        
        # Сортируем по дате
        all_results = sorted(all_results, key=lambda x: x['date'])
        
        # === 2. ОБЩАЯ СТАТИСТИКА ===
        total_predictions = len(all_results)
        total_days = sum(r['summary']['total_days'] for r in all_results)
        correct_directions = sum(r['summary']['correct_directions'] for r in all_results)
        overall_accuracy = (correct_directions / total_days) * 100 if total_days > 0 else 0
        
        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])
        avg_mape = np.mean([
            r['mape'] if r['mape'] <= 1000 else 0
            for r in all_results
        ])
        
        avg_mean_error = np.mean([r['mean_error'] for r in all_results])
        
        std_mae = np.std([r['mae'] for r in all_results])
        std_rmse = np.std([r['rmse'] for r in all_results])
        
        min_mae = min(r['mae'] for r in all_results)
        max_mae = max(r['mae'] for r in all_results)
        
        # === 3. ВЫВОД ОБЩЕЙ СТАТИСТИКИ ===
        print("\n" + "=" * 80)
        print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ПРОГНОЗАМ")
        print("=" * 80)
        print(f"📊 Период: {all_results[0]['date']} → {all_results[-1]['date']}")
        print(f"📈 Количество прогнозов: {total_predictions}")
        print(f"📅 Всего спрогнозировано дней: {total_days}")
        print(f"\n🎯 ТОЧНОСТЬ НАПРАВЛЕНИЯ:")
        print(f"   ├─ Правильных: {correct_directions} из {total_days}")
        print(f"   └─ Overall Direction Accuracy: {overall_accuracy:.2f}%")
        print(f"\n📉 ОШИБКИ:")
        print(f"   ├─ Средняя MAE:  {avg_mae:.6f} ± {std_mae:.6f}")
        print(f"   ├─ Средняя RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
        print(f"   ├─ Средняя MAPE: {avg_mape:.2f}%")
        print(f"   ├─ Min MAE:      {min_mae:.6f}")
        print(f"   ├─ Max MAE:      {max_mae:.6f}")
        print(f"   └─ Систем. ошибка: {avg_mean_error:.6f}")
        
        # === 4. СТАТИСТИКА ПО МОДЕЛЯМ ===
        unique_models = set(r.get('model_type', 'Unknown') for r in all_results)
        unique_feature_counts = set(r.get('num_features', 0) for r in all_results)
        
        print(f"\n🔧 КОНФИГУРАЦИЯ:")
        print(f"   ├─ Модели: {', '.join(unique_models)}")
        print(f"   ├─ Количество features: {', '.join(map(str, sorted(unique_feature_counts)))}")
        print(f"   └─ Target: {all_results[0].get('target', 'Unknown')}")
        print("=" * 80)
        
        # === 5. ДЕТАЛЬНАЯ ТАБЛИЦА ПО ДНЯМ ===
        print("\n" + "=" * 100)
        print("ДЕТАЛЬНАЯ ИСТОРИЯ ПО ДНЯМ")
        print("=" * 100)
        print(f"{'Дата':<12} {'Dir Acc':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Mean Err':<12} {'Features':<10}")
        print("-" * 100)
        
        for r in all_results:
            acc = r['direction_accuracy']
            status = '✅' if acc >= 50 else '❌'
            date = r['date']
            mae_val = r['mae']
            rmse_val = r['rmse']
            mape_val = 0 if r['mape'] > 1000 else r['mape']
            mean_err = r['mean_error']
            num_feat = r.get('num_features', 'N/A')
            
            print(f"{date:<12} {status} {acc:>5.1f}%  {mae_val:>10.6f}  {rmse_val:>10.6f}  {mape_val:>7.2f}%  {mean_err:>10.6f}  {num_feat:>8}")
        
        print("=" * 100)
        
        # === 6. СОЗДАЁМ DATAFRAME ===
   
        df_results = pd.DataFrame([{
            'date': r['date'],
            'direction_accuracy': r['direction_accuracy'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'mape': r['mape'],
            'mean_error': r['mean_error'],
            'num_features': r.get('num_features', 0),
            'model_type': r.get('model_type', 'Unknown'),
            # ✅ ADD THESE TWO LINES:
            'predictions': r['predictions'][0] if isinstance(r['predictions'], list) else r['predictions'],
            'real_values': r['real_values'][0] if isinstance(r['real_values'], list) else r['real_values']
        } for r in all_results])

        df_results['date'] = pd.to_datetime(df_results['date'])
        
        
        # Your plotting code
        dates = pd.to_datetime(df_results['date'])
        
        # Prepare data series
        real_series = df_results['real_values']
        pred_series = df_results['predictions']

        plt.figure(figsize=(16, 6))

        # 1. Plot Reality (Solid Line) - Acts as your "History"
        plt.plot(dates, real_series, 
                 label="Реальность (Fact)", 
                 color='steelblue', 
                 linewidth=2, 
                 linestyle='-')

        # 2. Plot Predictions (Dashed Line) - Acts as your "Forecast"
        plt.plot(dates, pred_series, 
                 label="Прогноз AI", 
                 color='tab:red', 
                 linewidth=2, 
                 linestyle='--', 
                 alpha=0.9)

        # 3. Highlight differences (Optional but looks like the screenshot)
        # Fill area between lines to show error magnitude
        plt.fill_between(dates, real_series, pred_series, 
                         color='gray', alpha=0.1, label='Error Gap')

        plt.title(f"Walk-Forward Validation: {len(df_results)} Days", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Target Value")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        
        # Format X-Axis to look nice
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate() # Rotate dates slightly

        plt.tight_layout()

        # Save
        save_path = folder if folder else './results'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/predictions_vs_reality.png', dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {save_path}/predictions_vs_reality.png")

        plt.show()
        
        # === 7. ВИЗУАЛИЗАЦИЯ ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Direction Accuracy по дням
        ax1 = axes[0, 0]
        colors = ['green' if x >= 50 else 'red' for x in df_results['direction_accuracy']]
        ax1.bar(range(len(df_results)), df_results['direction_accuracy'], color=colors, alpha=0.7)
        ax1.axhline(y=50, color='blue', linestyle='--', label='50% (Random)')
        ax1.axhline(y=overall_accuracy, color='orange', linestyle='--', label=f'Average: {overall_accuracy:.1f}%')
        ax1.set_xlabel('Прогноз #')
        ax1.set_ylabel('Direction Accuracy (%)')
        ax1.set_title('Direction Accuracy по дням')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: MAE по дням
        ax2 = axes[0, 1]
        ax2.plot(range(len(df_results)), df_results['mae'], marker='o', color='purple', label='MAE')
        ax2.axhline(y=avg_mae, color='red', linestyle='--', label=f'Average: {avg_mae:.6f}')
        ax2.fill_between(range(len(df_results)),
                        avg_mae - std_mae, avg_mae + std_mae,
                        alpha=0.2, color='red', label='±1 STD')
        ax2.set_xlabel('Прогноз #')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE по дням')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Систематическая ошибка (bias)
        ax3 = axes[1, 0]
        ax3.bar(range(len(df_results)), df_results['mean_error'],
                color=['red' if x > 0 else 'green' for x in df_results['mean_error']], alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=avg_mean_error, color='blue', linestyle='--', label=f'Average: {avg_mean_error:.6f}')
        ax3.set_xlabel('Прогноз #')
        ax3.set_ylabel('Mean Error (Bias)')
        ax3.set_title('Систематическая ошибка (Прогноз - Реальность)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # График 4: Cumulative Direction Accuracy
        ax4 = axes[1, 1]
        cumulative_correct = np.cumsum([r['summary']['correct_directions'] for r in all_results])
        cumulative_total = np.cumsum([r['summary']['total_days'] for r in all_results])
        cumulative_accuracy = (cumulative_correct / cumulative_total) * 100
        ax4.plot(range(len(df_results)), cumulative_accuracy, marker='o', color='darkgreen', linewidth=2)
        ax4.axhline(y=50, color='blue', linestyle='--', label='50% (Random)')
        ax4.fill_between(range(len(df_results)), 50, cumulative_accuracy,
                        where=(cumulative_accuracy >= 50), alpha=0.3, color='green', label='Above random')
        ax4.fill_between(range(len(df_results)), 50, cumulative_accuracy,
                        where=(cumulative_accuracy < 50), alpha=0.3, color='red', label='Below random')
        ax4.set_xlabel('Прогноз #')
        ax4.set_ylabel('Cumulative Direction Accuracy (%)')
        ax4.set_title('Накопительная точность направления')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.plot()
        
        plt.tight_layout()
        
        # ✅ Save plot
        if folder:
            plt.savefig(f'{folder}/analysis_summary.png', dpi=150, bbox_inches='tight')
            print(f"\n✅ График сохранён: {folder}/analysis_summary.png")
        else:
            plt.savefig('./analysis_summary.png', dpi=150, bbox_inches='tight')
            print(f"\n✅ График сохранён: ./analysis_summary.png")
        
        plt.show()
        
        # === 8. СОХРАНЕНИЕ В CSV ===
        csv_path = f'{folder}/analysis_summary.csv' if folder else './analysis_summary.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Таблица сохранена: {csv_path}")
        
        # === 9. СОХРАНЕНИЕ ОТЧЁТА В MD ===
        report = f"""# Отчёт по прогнозам

    **Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Период:** {all_results[0]['date']} → {all_results[-1]['date']}

    ## Общая статистика

    | Метрика | Значение |
    |---------|----------|
    | Количество прогнозов | {total_predictions} |
    | Всего дней | {total_days} |
    | Правильных направлений | {correct_directions} / {total_days} |
    | **Overall Direction Accuracy** | **{overall_accuracy:.2f}%** |
    | Средняя MAE | {avg_mae:.6f} ± {std_mae:.6f} |
    | Средняя RMSE | {avg_rmse:.6f} ± {std_rmse:.6f} |
    | Средняя MAPE | {avg_mape:.2f}% |
    | Систематическая ошибка | {avg_mean_error:.6f} |

    ## Детальная история

  
    | Дата | Direction Acc | Prediction | Reality | MAE | RMSE | MAPE | Mean Error | Features |
    |------|---------------|------------|---------|-----|------|------|------------|----------|
    """

        for r in all_results:
            acc = r['direction_accuracy']
            status = '✅' if acc >= 50 else '❌'
            
            # Extract prediction and reality
            pred = r['predictions'][0] if isinstance(r['predictions'], list) else r['predictions']
            real = r['real_values'][0] if isinstance(r['real_values'], list) else r['real_values']
            
            report += f"| {r['date']} | {status} {acc:.1f}% | {pred:.4f} | {real:.4f} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['mape']:.2f}% | {r['mean_error']:.6f} | {r.get('num_features', 'N/A')} |\n"

        # ✅ Add visualizations (handle None folder)
        report += f"\n## Visualizations\n\n"

        if folder:
            report += f"![Analysis Summary]({folder}/analysis_summary.png)\n\n"
            report += f"![Predictions vs Reality]({folder}/predictions_vs_reality.png)\n"
        else:
            report += f"![Analysis Summary](./analysis_summary.png)\n\n"
            report += f"![Predictions vs Reality](./predictions_vs_reality.png)\n"

        # ✅ Set report path (handle None folder)
        report_path = f'{folder}/REPORT.md' if folder else './REPORT.md'

        # ✅ Create directory if needed
        if folder:
            os.makedirs(folder, exist_ok=True)

        # ✅ Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ Отчёт сохранён: {report_path}")

        return df_results, all_results

            


class CustomForecaster:
    def __init__(self, base_pipeline, param_grid=None, horizon=7, cv_splits=3,one_time_grid_search=True):
        """
        base_pipeline: пайплайн (Scaler + Model)
        param_grid: сетка параметров для GridSearch
        horizon: горизонт прогнозирования
        cv_splits: количество фолдов для TimeSeriesSplit
        """
        self.base_pipeline = base_pipeline
        self.param_grid = param_grid
        self.horizon = horizon
        self.cv_splits = cv_splits
        self.one_time_grid_search = one_time_grid_search    

        # Словарь для хранения лучших моделей по шагам: {1: model_step_1, 2: model_step_2...}
        self.models = {}
        # Словарь для хранения истории лучших скоров (опционально)
        self.results = {}

    def train(self, X, y):
        print(f"Начинаем обучение {self.horizon} моделей с подбором гиперпараметров...")

        for step in range(1, self.horizon + 1):
            # 1. Формируем таргет для конкретного шага (сдвиг назад)
            y_shifted = y.shift(-step)

            # 2. Убираем NaN в конце (данные, для которых будущее неизвестно)
            valid_indices = y_shifted.dropna().index
            X_subset = X.loc[valid_indices]
            y_subset = y_shifted.loc[valid_indices]

            # 3. Настраиваем кросс-валидацию для временных рядов
            # Это важно! TimeSeriesSplit гарантирует, что train всегда в прошлом, а test в будущем
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)

            # 4. Создаем GridSearchCV
            # Важно: clone создает чистую копию пайплайна
            if self.param_grid is not None:
            
                grid = GridSearchCV(
                    estimator=clone(self.base_pipeline),
                    param_grid=self.param_grid,
                    scoring="neg_mean_absolute_percentage_error", # Ваша метрика
                    cv=tscv,
                    n_jobs=-1,
                    verbose=1 # Можно поставить 1, чтобы видеть прогресс внутри шага
                )
                # 5. Запускаем поиск лучших параметров для ТЕКУЩЕГО шага (step)
                grid.fit(X_subset, y_subset)

                # 6. Сохраняем лучшую модель (она уже обучена на всем X_subset внутри grid.fit)
                self.models[step] = grid.best_estimator_
                self.results[step] = grid.best_score_

                print(f"Шаг {step}/{self.horizon} готов. Лучший Score (MAPE): {grid.best_score_:.4f}")
                # Можно вывести параметры, если интересно:
                # print(f"Best params step {step}: {grid.best_params_}")
                print("Train ist fertig mit GridSearchCV.")
                
                
            elif self.param_grid is not None and self.one_time_grid_search:
                pass
            
            else:
                
                model = clone(self.base_pipeline)
                model.fit(X_subset, y_subset)

                self.models[step] = model
                self.results[step] = None
                
                print(f"Шаг {step}/{self.horizon} готов (без GridSearch).")
                   
    def predict_future(self, X_current):
        """
        Прогноз от последней точки данных (X_current - это 1 строка DataFrame/Series)
        """
        forecasts = []
        days = []

        # X_current может прийти как Series, преобразуем в DataFrame (транспонируем)
        if isinstance(X_current, pd.Series):
            X_current = X_current.to_frame().T

        for step in range(1, self.horizon + 1):
            model = self.models[step]
            # Делаем предикт
            pred = model.predict(X_current)[0]
            ### How to realize it 
                          
            forecasts.append(pred)
            days.append(step)

        return pd.DataFrame({'step': days, 'prediction': forecasts})
    

class OptunaForecaster:
    def __init__(self, base_pipeline, param_grid=None, horizon=7, cv_splits=3,one_time_grid_search=True):
        """
        base_pipeline: пайплайн (Scaler + Model)
        param_grid: сетка параметров для GridSearch
        horizon: горизонт прогнозирования
        cv_splits: количество фолдов для TimeSeriesSplit
        """
        self.base_pipeline = base_pipeline
        self.param_grid = param_grid
        self.horizon = horizon
        self.cv_splits = cv_splits
        self.one_time_grid_search = one_time_grid_search    

        # Словарь для хранения лучших моделей по шагам: {1: model_step_1, 2: model_step_2...}
        self.models = {}
        # Словарь для хранения истории лучших скоров (опционально)
        self.results = {}

    def train(self, X, y):
        print(f"Начинаем обучение {self.horizon} моделей с подбором гиперпараметров...")

        for step in range(1, self.horizon + 1):
            # 1. Формируем таргет для конкретного шага (сдвиг назад)
            y_shifted = y.shift(-step)

            # 2. Убираем NaN в конце (данные, для которых будущее неизвестно)
            valid_indices = y_shifted.dropna().index
            X_subset = X.loc[valid_indices]
            y_subset = y_shifted.loc[valid_indices]

            # 3. Настраиваем кросс-валидацию для временных рядов
            # Это важно! TimeSeriesSplit гарантирует, что train всегда в прошлом, а test в будущем
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)

            # 4. Создаем GridSearchCV
            # Важно: clone создает чистую копию пайплайна
            if self.param_grid is not None:
            
                grid = GridSearchCV(
                    estimator=clone(self.base_pipeline),
                    param_grid=self.param_grid,
                    scoring="neg_mean_absolute_percentage_error", # Ваша метрика
                    cv=tscv,
                    n_jobs=-1,
                    verbose=1 # Можно поставить 1, чтобы видеть прогресс внутри шага
                )
                # 5. Запускаем поиск лучших параметров для ТЕКУЩЕГО шага (step)
                grid.fit(X_subset, y_subset)

                # 6. Сохраняем лучшую модель (она уже обучена на всем X_subset внутри grid.fit)
                self.models[step] = grid.best_estimator_
                self.results[step] = grid.best_score_

                print(f"Шаг {step}/{self.horizon} готов. Лучший Score (MAPE): {grid.best_score_:.4f}")
                # Можно вывести параметры, если интересно:
                # print(f"Best params step {step}: {grid.best_params_}")
                print("Train ist fertig mit GridSearchCV.")
                
                
            elif self.param_grid is not None and self.one_time_grid_search:
                pass
            
            else:
                
                model = clone(self.base_pipeline)
                model.fit(X_subset, y_subset)

                self.models[step] = model
                self.results[step] = None
                
                print(f"Шаг {step}/{self.horizon} готов (без GridSearch).")
                   
    def predict_future(self, X_current):
        """
        Прогноз от последней точки данных (X_current - это 1 строка DataFrame/Series)
        """
        forecasts = []
        days = []

        # X_current может прийти как Series, преобразуем в DataFrame (транспонируем)
        if isinstance(X_current, pd.Series):
            X_current = X_current.to_frame().T

        for step in range(1, self.horizon + 1):
            model = self.models[step]
            # Делаем предикт
            pred = model.predict(X_current)[0]
            ### How to realize it 
                          
            forecasts.append(pred)
            days.append(step)

        return pd.DataFrame({'step': days, 'prediction': forecasts})
    
    


class BaselineForecaster:
    def __init__(self, horizon=7):
        self.models = {} # Словарь для хранения 7 обученных пайплайнов
        self.horizon = horizon   # Прогноз на 7 дней вперед

    def train(self, X_train, y_train):
        """
        Обучает 7 отдельных моделей.
        X_train: признаки (pandas DataFrame)
        y_train: целевая переменная (pandas Series) на текущий день
        """
        print("Начинаем обучение 7 моделей...")

        for step in range(1, self.horizon + 1):
            # 1. Подготовка целевой переменной для конкретного дня (step)
            # Если step=1, мы учимся предсказывать t+1.
            # Если step=7, мы учимся предсказывать t+7.

            # Сдвигаем таргет назад, чтобы напротив X(t) стоял y(t+step)
            y_shifted = y_train.shift(-step)

            # 2. Выравнивание данных
            # Нам нужно удалить последние 'step' строк, так как у них нет будущего таргета (NaN)
            # Используем индексы из y_shifted, которые не NaN
            valid_indices = y_shifted.dropna().index

            X_subset = X_train.loc[valid_indices]
            y_subset = y_shifted.loc[valid_indices]

            # 3. Создание пайплайна: Скалер -> Регрессия
            # Pipeline сам обучит scaler на X_subset и применит его
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])

            # 4. Обучение
            pipe.fit(X_subset, y_subset)

            # Сохраняем модель
            self.models[step] = pipe
            print(f"Модель для дня +{step} обучена.")

    def predict(self, X_input):
        """
        Делает прогноз на 7 дней вперед на основе ВХОДНЫХ данных.
        X_input: признаки для текущего дня (одна строка или DataFrame).
        """
        forecasts = {}

        # Проходим по всем 7 моделям
        for step in range(1, self.horizon + 1):
            model = self.models[step]

            # Модель сама отскалирует входные данные (внутри pipe) и сделает прогноз
            pred = model.predict(X_input)

            # Если подали одну строку, берем значение, если много - возвращаем массив
            if len(pred) == 1:
                forecasts[f'Day_{step}'] = pred[0]
            else:
                forecasts[f'Day_{step}'] = pred

        return forecasts
    
    
    

    