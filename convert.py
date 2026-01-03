import pandas as pd
from pathlib import Path

CSV_DIR = Path("data/market")
PARQUET_DIR = Path("data/ohlc")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

def convert_csv_to_parquet():
    csv_files = list(CSV_DIR.glob("*.csv"))
    print(f"Найдено {len(csv_files)} CSV файлов")

    for csv_file in csv_files:
        try:
            path = csv_file.relative_to(CSV_DIR)
            df = pd.read_csv(CSV_DIR / path, sep=";")

            # timestamp (ms) → datetime UTC
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # канонический порядок колонок
            cols = ['datetime'] + [col for col in df.columns if col != 'timestamp' and col != 'datetime']
            df = df[cols]

            # обязательные условия
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            df.to_parquet(
                PARQUET_DIR / path.with_suffix(".parquet"),
                engine="pyarrow",
                compression="zstd"
            )
        except Exception as e:
            print(f"Ошибка при конвертации {csv_file}: {e}")
            continue

    print(f"Saved {PARQUET_DIR}")

if __name__ == "__main__":
    convert_csv_to_parquet()