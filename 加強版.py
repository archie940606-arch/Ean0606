import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance as yf

# 關閉不安全的 HTTPS 警告與全域底層警告 Log
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ----------------------------------------------------------------------
# 1. 自動擷取全台股清單與中文名稱 (TWSE / TPEx 官方 API)
# ----------------------------------------------------------------------
def get_all_taiwan_tickers():
    print(
        "🌐 正在透過證交所與櫃買中心官方 API 擷取全台股股票/ETF"
        " 清單與名稱..."
    )
    ticker_map = {}  # 格式: {'2330.TW': '台積電', ...}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # A. 抓取上市股票與名稱 (TWSE 官方 API)
    try:
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        res = requests.get(twse_url, headers=headers, timeout=10, verify=False)
        if res.status_code == 200:
            data = res.json()
            for row in data:
                code = str(row.get("Code", "")).strip()
                name = str(row.get("Name", "")).strip()
                if code.isdigit() and (len(code) == 4 or len(code) in [5, 6]):
                    ticker_map[f"{code}.TW"] = name
            twse_count = len(
                [t for t in ticker_map.keys() if t.endswith(".TW")]
            )
            print(f"  └─ 已成功載入上市標的: {twse_count} 檔")
    except Exception as e:
        print(f"⚠️ 抓取上市清單失敗: {e}")

    # B. 抓取上櫃股票與名稱 (TPEx 官方 API)
    try:
        tpex_url = "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&o=json"
        res = requests.get(tpex_url, headers=headers, timeout=10, verify=False)
        if res.status_code == 200:
            data = res.json()
            aaData = data.get("aaData", [])
            tpex_count = 0
            for row in aaData:
                if isinstance(row, list) and len(row) > 1:
                    code = str(row[0]).strip()
                    name = str(row[1]).strip()
                    if code.isdigit() and (len(code) == 4 or len(code) in [5, 6]):
                        ticker_map[f"{code}.TWO"] = name
                        tpex_count += 1
            print(f"  └─ 已成功載入上櫃標的: {tpex_count} 檔")
    except Exception as e:
        print(f"⚠️ 抓取上櫃清單失敗: {e}")

    if not ticker_map:
        print("⚠️ 無法取得 API 清單，切換至備用清單...")
        backup_list = [
            ("2330.TW", "台積電"),
            ("2317.TW", "鴻海"),
            ("2454.TW", "聯發科"),
            ("2308.TW", "台達電"),
            ("2382.TW", "廣達"),
            ("2303.TW", "聯電"),
            ("2881.TW", "富邦金"),
            ("2882.TW", "國泰金"),
            ("0050.TW", "元大台灣50"),
            ("0056.TW", "元大高股息"),
            ("00878.TW", "國泰永續高股息"),
            ("00919.TW", "群益台灣精選高息"),
        ]
        ticker_map = dict(backup_list)
    else:
        print(f"✅ 成功擷取全台股共 {len(ticker_map)} 檔標的！\n")

    return ticker_map


# ----------------------------------------------------------------------
# 2. 股價位階判斷 (底部起漲 vs 高檔末升段)
# ----------------------------------------------------------------------
def check_breakout_stage(
    close_s, high_s, low_s, filter_ema=200, lookback_window=252
):
    latest_close = close_s.iloc[-1]

    # 年線乖離率 (%)
    ema200 = close_s.ewm(span=filter_ema, adjust=False).mean().iloc[-1]
    ema_bias = ((latest_close - ema200) / ema200) * 100

    # 近 1 年最低點起的累計漲幅 (%)
    period_low = low_s.iloc[-lookback_window:].min()
    gain_from_low = ((latest_close - period_low) / period_low) * 100

    # 近 5 天 20/60/120/200 均線糾結度 (標準差 / 平均值，取近 5 天平均)
    ma20 = close_s.rolling(20).mean()
    ma60 = close_s.rolling(60).mean()
    ma120 = close_s.rolling(120).mean()
    ma200 = close_s.rolling(200).mean()

    ma_df = pd.concat([ma20, ma60, ma120, ma200], axis=1).iloc[-5:]
    ma_compression = (
        (ma_df.std(axis=1) / ma_df.mean(axis=1)) * 100
    ).mean()

    # 綜合位階判定
    if ema_bias > 30.0 or gain_from_low > 80.0:
        stage = "⚠️高檔末升段"
        stage_code = "HIGH_RISK"
    elif ema_bias <= 15.0 and gain_from_low <= 40.0 and ma_compression < 3.5:
        stage = "🔥底部起漲"
        stage_code = "BOTTOM_START"
    else:
        stage = "📈溫和主升段"
        stage_code = "MID_TREND"

    return {
        "位階標籤": stage,
        "位階代碼": stage_code,
        "年線乖離(%)": round(ema_bias, 1),
        "距低點漲幅(%)": round(gain_from_low, 1),
    }


# ----------------------------------------------------------------------
# 3. Mark Minervini VCP 波動收縮型態檢測
# ----------------------------------------------------------------------
def detect_vcp_pattern(close_s, high_s, low_s, vol_s):
    if len(close_s) < 252:
        return False, {}

    latest_close = close_s.iloc[-1]

    # SEPA 趨勢模版過濾
    sma50 = close_s.rolling(50).mean().iloc[-1]
    sma150 = close_s.rolling(150).mean().iloc[-1]
    sma200 = close_s.rolling(200).mean().iloc[-1]
    sma200_20d_ago = close_s.rolling(200).mean().iloc[-20]

    high_52w = high_s.iloc[-252:].max()
    dist_from_52w_high = (high_52w - latest_close) / high_52w

    trend_pass = (
        (latest_close > sma150)
        and (latest_close > sma200)
        and (sma50 > sma150)
        and (sma150 > sma200)
        and (sma200 > sma200_20d_ago)
        and (dist_from_52w_high <= 0.25)
    )

    if not trend_pass:
        return False, {}

    # 三段收縮測試 (近 60 天切分為 3 個 20 天)
    seg1_high, seg1_low = high_s.iloc[-60:-40].max(), low_s.iloc[-60:-40].min()
    depth1 = (seg1_high - seg1_low) / seg1_high if seg1_high > 0 else 0

    seg2_high, seg2_low = high_s.iloc[-40:-20].max(), low_s.iloc[-40:-20].min()
    depth2 = (seg2_high - seg2_low) / seg2_high if seg2_high > 0 else 0

    seg3_high, seg3_low = high_s.iloc[-20:].max(), low_s.iloc[-20:].min()
    depth3 = (seg3_high - seg3_low) / seg3_high if seg3_high > 0 else 0

    is_contraction = (depth1 > depth2) and (depth2 > depth3) and (depth3 <= 0.10)

    # 量能乾枯測試 (VDU)
    vol_5d_avg = vol_s.iloc[-5:].mean()
    vol_50d_avg = vol_s.iloc[-50:].mean()
    is_vdu = (vol_5d_avg / vol_50d_avg) <= 0.65 if vol_50d_avg > 0 else False

    is_vcp = trend_pass and is_contraction and is_vdu

    metrics = {
        "VCP樞紐買點": round(seg3_high, 2),
        "末端收縮(%)": round(depth3 * 100, 1),
        "量能縮減比": round(vol_5d_avg / vol_50d_avg, 2),
    }

    return is_vcp, metrics


# ----------------------------------------------------------------------
# 4. 單批次資料處理與指標計算 (線程安全版本)
# ----------------------------------------------------------------------
def process_batch(
    batch,
    ticker_map,
    min_avg_volume_shares,
    entry_window,
    exit_window,
    filter_ema,
):
    buy_signals = []
    vcp_signals = []
    sell_signals = []
    latest_date = "未知"

    try:
        # 直接下載資料，不進行 contextlib 輸出流重定向，確保多執行緒穩定
        # 加入簡單重試機制，避免暫時性限流/網路錯誤導致整批直接放棄
        data = pd.DataFrame()
        last_err = None
        for attempt in range(3):
            try:
                data = yf.download(
                    batch, period="1y", progress=False, threads=False
                )
                if not data.empty:
                    break
            except Exception as e:
                last_err = e
            time.sleep(1.5 * (attempt + 1))

        if data.empty:
            if last_err is not None:
                print(f"⚠️ 批次下載失敗（已重試 3 次）: {batch[:3]}... 錯誤: {last_err}")
            return buy_signals, vcp_signals, sell_signals, latest_date

        for ticker in batch:
            try:
                if len(batch) > 1:
                    if ("Close", ticker) in data.columns:
                        sub = pd.concat(
                            {
                                "Open": data["Open"][ticker],
                                "High": data["High"][ticker],
                                "Low": data["Low"][ticker],
                                "Close": data["Close"][ticker],
                                "Volume": data["Volume"][ticker],
                            },
                            axis=1,
                        ).dropna(how="any")
                    else:
                        continue
                else:
                    sub = pd.concat(
                        {
                            "Open": data["Open"],
                            "High": data["High"],
                            "Low": data["Low"],
                            "Close": data["Close"],
                            "Volume": data["Volume"],
                        },
                        axis=1,
                    ).dropna(how="any")

                if sub.empty:
                    continue

                # 統一從同一份已對齊日期索引的 sub 取出各欄位，
                # 避免各欄位各自 dropna 造成日期錯位
                open_s = sub["Open"]
                high_s = sub["High"]
                low_s = sub["Low"]
                close_s = sub["Close"]
                vol_s = sub["Volume"]

                if len(close_s) < filter_ema:
                    continue

                latest_date = close_s.index[-1].strftime("%Y-%m-%d")
                stock_name = ticker_map.get(ticker, "未知")

                # 流動性過濾
                avg_vol_20 = vol_s.iloc[-20:].mean()
                if avg_vol_20 < min_avg_volume_shares:
                    continue

                # 基礎價格與動態通道
                latest_open = open_s.iloc[-1]
                latest_close = close_s.iloc[-1]
                latest_high = high_s.iloc[-1]
                latest_low = low_s.iloc[-1]
                latest_vol = vol_s.iloc[-1]

                upper_channel = (
                    high_s.shift(1).rolling(window=entry_window).max().iloc[-1]
                )
                lower_channel = (
                    low_s.shift(1).rolling(window=exit_window).min().iloc[-1]
                )
                ema200 = (
                    close_s.ewm(span=filter_ema, adjust=False).mean().iloc[-1]
                )

                # 量比與 K 線防倒貨過濾條件
                vol_ratio = (
                    round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0
                )
                candle_range = latest_high - latest_low
                close_location = (
                    (latest_close - latest_low) / candle_range
                    if candle_range > 0
                    else 1
                )

                # 防倒貨濾網：1. 上影線不長(收高於65%區域) 2. 不開高走低黑棒 3. 不暴放大於5倍極端天量
                is_clean_breakout = (
                    (close_location >= 0.65)
                    and (latest_close >= latest_open)
                    and (vol_ratio <= 5.0)
                )

                # 計算 14 日 ATR 與風控停損價
                tr = pd.concat(
                    [
                        high_s - low_s,
                        (high_s - close_s.shift(1)).abs(),
                        (low_s - close_s.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr14 = tr.rolling(window=14).mean().iloc[-1]
                suggested_stop_loss = round(latest_close - (2 * atr14), 2)

                # 股價位階檢查
                stage_info = check_breakout_stage(close_s, high_s, low_s)

                # VCP 型態檢查
                is_vcp, vcp_metrics = detect_vcp_pattern(
                    close_s, high_s, low_s, vol_s
                )

                # ------------------------------------------------------
                # 訊號觸發條件分類
                # ------------------------------------------------------
                # A. 經典海龜突破 (且剔除倒貨與高檔末升段)
                if (
                    latest_close > upper_channel
                    and latest_close > ema200
                    and is_clean_breakout
                    and stage_info["位階代碼"] != "HIGH_RISK"
                ):

                    buy_signals.append(
                        {
                            "代碼": ticker.split(".")[0],
                            "股票名稱": stock_name,
                            "最新收盤": round(latest_close, 2),
                            "突破點": round(upper_channel, 2),
                            "股價位階": stage_info["位階標籤"],
                            "量比(倍)": vol_ratio,
                            "年線乖離(%)": stage_info["年線乖離(%)"],
                            "建議停損(2*ATR)": suggested_stop_loss,
                            "20日均量(張)": int(avg_vol_20 / 1000),
                            "資料日期": latest_date,
                        }
                    )

                # B. VCP 籌碼乾枯樞紐買進清單
                if is_vcp and latest_close >= vcp_metrics["VCP樞紐買點"]:
                    vcp_signals.append(
                        {
                            "代碼": ticker.split(".")[0],
                            "股票名稱": stock_name,
                            "最新收盤": round(latest_close, 2),
                            "VCP樞紐價": vcp_metrics["VCP樞紐買點"],
                            "末端收縮(%)": vcp_metrics["末端收縮(%)"],
                            "量縮乾枯比": vcp_metrics["量能縮減比"],
                            "建議停損(2*ATR)": suggested_stop_loss,
                            "20日均量(張)": int(avg_vol_20 / 1000),
                            "資料日期": latest_date,
                        }
                    )

                # C. 跌破 10 日低點賣出訊號
                elif latest_close < lower_channel:
                    sell_signals.append(
                        {
                            "代碼": ticker.split(".")[0],
                            "股票名稱": stock_name,
                            "最新收盤": round(latest_close, 2),
                            "10日跌破點": round(lower_channel, 2),
                            "20日均量(張)": int(avg_vol_20 / 1000),
                            "資料日期": latest_date,
                        }
                    )

            except Exception as e:
                print(f"⚠️ 個股處理失敗 [{ticker}]: {e}")
                continue

    except Exception as e:
        print(f"⚠️ 批次處理發生例外: {e}")

    return buy_signals, vcp_signals, sell_signals, latest_date


# ----------------------------------------------------------------------
# 5. 主執行引擎 (多執行緒)
# ----------------------------------------------------------------------
def run_post_market_scanner(
    min_avg_volume_shares=300_000,
    entry_window=20,
    exit_window=10,
    filter_ema=200,
    batch_size=25,
    max_workers=8,
):
    ticker_map = get_all_taiwan_tickers()
    all_tickers = list(ticker_map.keys())

    triggered_buy = []
    triggered_vcp = []
    triggered_sell = []
    latest_data_date = "未知"

    batches = [
        all_tickers[i : i + batch_size]
        for i in range(0, len(all_tickers), batch_size)
    ]
    total_batches = len(batches)

    print(
        f"🚀 開始執行【全市場多重篩選定量掃描】（線程數: {max_workers} | 總批次:"
        f" {total_batches}）..."
    )

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_batch,
                batch,
                ticker_map,
                min_avg_volume_shares,
                entry_window,
                exit_window,
                filter_ema,
            ): batch
            for batch in batches
        }

        for future in as_completed(futures):
            completed_count += 1
            buys, vcps, sells, b_date = future.result()
            triggered_buy.extend(buys)
            triggered_vcp.extend(vcps)
            triggered_sell.extend(sells)
            if b_date != "未知":
                latest_data_date = b_date

            if completed_count % 10 == 0 or completed_count == total_batches:
                print(
                    f"  └─ 進度: {completed_count}/{total_batches} 批次完成 |"
                    f" 當前日期: {latest_data_date} | 累計突破買進:"
                    f" {len(triggered_buy)} 檔 / VCP: {len(triggered_vcp)} 檔"
                )

    return (
        pd.DataFrame(triggered_buy),
        pd.DataFrame(triggered_vcp),
        pd.DataFrame(triggered_sell),
        latest_data_date,
    )


# ----------------------------------------------------------------------
# 6. 主程式執行與報表匯出
# ----------------------------------------------------------------------
if __name__ == "__main__":
    buy_df, vcp_df, sell_df, data_date = run_post_market_scanner(
        min_avg_volume_shares=300_000,  # 流動性門檻: 300 張
        entry_window=20,
        exit_window=10,
        filter_ema=200,
        batch_size=25,
        max_workers=8,
    )

    print("\n" + "=" * 90)
    print(
        f"📊 全台股量化選股進階掃描報告 | 資料基準日: 【 {data_date} 】"
    )
    print("=" * 90)

    print(
        "\n🔥【海龜突破 / 實戰優選買進清單】（剔除假突破倒貨與高檔末升段）："
    )
    if not buy_df.empty:
        buy_df = buy_df.sort_values(by="量比(倍)", ascending=False)
        print(buy_df.to_string(index=False))
    else:
        print("  └─ 今日全市場無符合海龜突破之優良標的。")

    print("\n⭐【VCP 波動收縮樞紐買進清單】（Minervini 籌碼沉澱突破）：")
    if not vcp_df.empty:
        vcp_df = vcp_df.sort_values(by="末端收縮(%)", ascending=True)
        print(vcp_df.to_string(index=False))
    else:
        print("  └─ 今日全市場無符合 VCP 型態之標的。")

    print("\n⚠️【跌破賣出/平倉訊號清單】（跌破 10 日低點）：")
    if not sell_df.empty:
        sell_df = sell_df.sort_values(by="20日均量(張)", ascending=False)
        print(sell_df.to_string(index=False))
    else:
        print("  └─ 今日全市場無符合賣出條件之股票。")

    print("\n" + "=" * 90)

    # 自動選擇 Excel 或 CSV 格式導出
    # 若整批下載都失敗導致 data_date 仍是「未知」，改用今天日期避免檔名異常
    safe_date = (
        data_date if data_date != "未知" else datetime.now().strftime("%Y-%m-%d")
    )
    filename = f"量化掃描報告_{safe_date.replace('-', '')}.xlsx"
    try:
        with pd.ExcelWriter(filename) as writer:
            if not buy_df.empty:
                buy_df.to_excel(
                    writer, sheet_name="海龜突破買進", index=False
                )
            if not vcp_df.empty:
                vcp_df.to_excel(
                    writer, sheet_name="VCP樞紐買進", index=False
                )
            if not sell_df.empty:
                sell_df.to_excel(
                    writer, sheet_name="跌破賣出清單", index=False
                )
        print(f"💾 掃描結果已成功儲存至 Excel 檔案: 【 {filename} 】")
    except Exception as e:
        print(f"⚠️ Excel 匯出失敗，改用 CSV 備援匯出: {e}")
        if not buy_df.empty:
            buy_df.to_csv(
                f"突破買進_{safe_date}.csv", index=False, encoding="utf-8-sig"
            )
            print(f"💾 買進清單已匯出至: 突破買進_{safe_date}.csv")
        if not vcp_df.empty:
            vcp_df.to_csv(
                f"VCP樞紐買進_{safe_date}.csv", index=False, encoding="utf-8-sig"
            )
            print(f"💾 VCP清單已匯出至: VCP樞紐買進_{safe_date}.csv")
        if not sell_df.empty:
            sell_df.to_csv(
                f"跌破賣出_{safe_date}.csv", index=False, encoding="utf-8-sig"
            )
            print(f"💾 賣出清單已匯出至: 跌破賣出_{safe_date}.csv")