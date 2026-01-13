#!/usr/bin/env python3
"""
Binance 数据下载脚本 - 在可访问币安的服务器上运行

使用方法:
1. pip install ccxt pandas
2. python download_binance.py
3. 下载完成后将 CSV 文件传回本地
"""

import ccxt
import pandas as pd
import time
from datetime import datetime

def download_binance_ohlcv(
    symbol: str = "ETH/USDT",
    timeframe: str = "4h",
    start_date: str = "2019-01-01",
    end_date: str = "2026-01-01",
):
    """下载币安 OHLCV 数据"""

    print("=" * 60)
    print("Binance Data Downloader")
    print("=" * 60)
    print(f"Symbol:     {symbol}")
    print(f"Timeframe:  {timeframe}")
    print(f"Start:      {start_date}")
    print(f"End:        {end_date}")
    print("=" * 60)

    # 初始化交易所
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    # 时间戳
    start_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")

    all_ohlcv = []
    current_ts = start_ts

    # 每次请求的 K 线数量
    limit = 1000

    # 时间间隔 (毫秒)
    timeframe_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    interval_ms = timeframe_ms.get(timeframe, 4 * 60 * 60 * 1000)

    print("\nDownloading...")

    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=current_ts,
                limit=limit,
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # 更新时间戳
            last_ts = ohlcv[-1][0]
            current_ts = last_ts + interval_ms

            # 进度
            current_date = datetime.utcfromtimestamp(last_ts / 1000).strftime("%Y-%m-%d")
            print(f"  Downloaded to: {current_date} ({len(all_ohlcv)} bars)")

            # 避免速率限制
            time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

    # 转换为 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # 转换时间戳
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 去重
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    # 过滤日期范围
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]

    # 保存
    symbol_clean = symbol.replace("/", "")
    filename = f"{symbol_clean}_{timeframe}_{start_date}_{end_date}.csv"
    df.to_csv(filename, index=False)

    print()
    print("=" * 60)
    print(f"Downloaded {len(df)} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {filename}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # 下载 ETH/USDT 4H 数据
    download_binance_ohlcv(
        symbol="ETH/USDT",
        timeframe="4h",
        start_date="2019-01-01",
        end_date="2026-01-01",
    )

    print("\n完成! 请将 CSV 文件传回本地使用。")
