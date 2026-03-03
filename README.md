# Trade & Orderbook Pipeline with Advanced Backtesting and Anomaly Detection

This project implements a comprehensive, real-time data pipeline for processing cryptocurrency exchange data (both Centralized and Decentralized), extracting advanced features, detecting anomalies, and backtesting trading strategies with detailed event-level analytics. It is designed for high-frequency market analysis and algorithmic trading research.

## Project Overview

The core objective of this project is to provide a robust and extensible framework for:
*   **Real-time Market Data Ingestion:** Connecting to various cryptocurrency exchanges (CEX and DEX) to stream live orderbook and trade data.
*   **Unified Data Normalization:** Standardizing diverse exchange data into a consistent internal format for seamless processing.
*   **Advanced Feature Engineering:** Generating a rich set of market microstructure, technical, and cross-exchange features.
*   **Anomaly Detection:** Identifying unusual patterns or events in real-time market data indicative of market manipulation or significant shifts.
*   **Historical Backtesting:** Simulating the entire pipeline against historical data to evaluate strategies and analyze market events.
*   **Event-Driven Analytics:** Providing granular, statistical insights into market behavior around specific events (e.g., pump/dump events).

## Architecture

The system is built with a modular and asynchronous architecture, primarily leveraging `FastAPI` for its web interface and `asyncio` for concurrent operations. Key components include:

*   **`trade_orderbook_pipeline.py`**: The main application entry point, orchestrating the entire pipeline. It sets up FastAPI endpoints for real-time data feeds and backtest execution.
*   **`core/exchange_manager_updated.py`**: Manages connections to CEX and DEX exchanges, handling dynamic adapter loading and callback mechanisms.
*   **`adapters/`**: Contains modular adapters for different exchange types. `cex_adapters.py` uses `CCXT` for CEX integration, while `dex_adapters.py` uses `Web3.py` for DEX interactions.
*   **`core/data_loader.py`**: Responsible for buffering, snapshotting, and aggregating normalized data from exchanges.
*   **`features/extractors/`**: A directory housing various specialized feature extractors. These modules take raw or aggregated market data and compute derived metrics (e.g., `AdvancedCryptofeedExtractor`, `AdvancedTradebookExtractor`, `EnhancedTechnicalExtractor`, `DEXLiquidityExtractor`, `FuturesOpenInterestFundingExtractor`, `CrossExchangeDiscrepancyExtractor`, `EnhancedOptionsExtractor`).
*   **`detection/system.py`**: Orchestrates multiple anomaly detection algorithms (`PyOD`, `OrderSkew`, `ChangePoint`, `HMM`, `MatrixProfile`, `LevelShift`) to identify unusual market conditions.
*   **`backtesting/engine.py`**: The core backtesting engine, which replays historical data through the feature extraction and anomaly detection pipeline, generating detailed event-level statistics.
*   **`core/gdrive_utils.py`**: Facilitates loading historical data for backtesting directly from Google Drive.
*   **`core/unified_output.py`**: Handles consistent output formatting and saving of processed data and results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/Order.git
    cd Order
    ```

2.  **Install dependencies:**
    The project relies on several Python libraries. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Google Drive Configuration (for Backtesting Historical Data):**
    Historical data for backtesting is loaded from Google Drive. You need to provide your Google Drive API key and the root folder IDs for your orderbook and tradebook data in the `BacktestConfig`.

    *   **Obtain a Google Drive API key:** Follow Google's documentation to create a project and enable the Google Drive API, then generate an API key.
    *   **Configure `BacktestConfig`:** In `config/pipeline_config.json` or `config/pipeline_config.yaml`, update the `gdrive_api_key` field with your obtained key.
    *   **Specify Root Folder IDs:**
        *   `gdrive_orderbook_root_id`: The Google Drive folder ID where your orderbook data is stored.
        *   `gdrive_tradebook_root_id`: The Google Drive folder ID for your tradebook data.
    *   **Data Organization:** Ensure your historical data on Google Drive follows a specific hierarchical structure:
        `ROOT_FOLDER_ID/SYMBOL_FOLDER/YEAR_FOLDER/MONTH_FOLDER/YYYY-MM-DD.csv.gz`
        (e.g., `1Azx57ItNc8d3BKPQeT1frPtxbY5ogaix/bnsusdt/2021/03/2021-03-27.csv.gz`)
        The `_load_gdrive_data` function in `backtesting/engine.py` expects this structure. Casing and delimiters for folder and file names are important.

## How to Use the Project

### 1. Starting the FastAPI Application

The project's core functionalities, including the real-time data pipeline and backtesting engine, are exposed via a FastAPI web server.

To start the server:

```bash
uvicorn trade_orderbook_pipeline:app --reload
```

This will typically start the server at `http://127.0.0.1:8000`. You can then access the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 2. Running Backtests

Once the FastAPI application is running, you can initiate backtests by sending a POST request to the `/backtest/run` endpoint.

**Example `curl` Command:**

```bash
curl -X POST "http://localhost:8000/backtest/run?pump_symbols=bnsusdt&control_symbols=gmtusdt&start_date=2021-03-27&end_date=2021-03-27&start_time=16%3A00%3A00&end_time=16%3A14%3A59&replay_speed=10&enable_features=true&enable_anomaly_detection=true&data_type=orderbook&enable_event_aggregation=true" \
     -H "Content-Type: application/json"
```

**Key Parameters for `/backtest/run` Endpoint:**

*   `pump_symbols` (str, comma-separated): Symbols identified as "pump" events (e.g., `bnsusdt`).
*   `control_symbols` (str, comma-separated): Symbols identified as "control" events (e.g., `gmtusdt`).
*   `start_date` (str, YYYY-MM-DD): The start date for the backtest period.
*   `end_date` (str, YYYY-MM-DD): The end date for the backtest period.
*   `start_time` (str, HH:MM:SS): The start time within each day for data loading.
*   `end_time` (str, HH:MM:SS): The end time within each day for data loading.
*   `replay_speed` (float): Multiplier for the simulation speed (e.g., `1.0` for real-time, `10.0` for 10x speed).
*   `enable_features` (bool): Set to `true` to enable feature extraction.
*   `enable_anomaly_detection` (bool): Set to `true` to enable anomaly detection.
*   `data_type` (str): Specifies the type of historical data to load: `orderbook`, `tradebook`, or `all`.
*   `enable_event_aggregation` (bool): Set to `true` to enable event-level aggregation and generate the `stats.jsonl` output with advanced diagnostics.

### 3. Interpreting `stats.jsonl` (Advanced Diagnostics)

When `enable_event_aggregation` is set to `true`, the backtesting engine generates a `stats_YYYYMMDD_HHMMSS.jsonl` file in the `backtest_results/` directory. This file contains event-level aggregated statistics, providing deep insights into market microstructure and anomaly characteristics around defined events. Each line in the file is a JSON object representing the aggregated statistics for a specific (symbol, event\_time, event\_type, window\_name) combination.

The output for each record includes:

*   **Basic Event Metadata:**
    *   `symbol`: The trading pair (e.g., `bnsusdt`).
    *   `event_time`: Timestamp of the event.
    *   `event_type`: `pump` or `control`.
    *   `window_name`: `pre_event` (e.g., -30 to 0 minutes before event) or `post_event` (e.g., 0 to +15 minutes after event).
    *   `window_start`, `window_end`: Actual start and end timestamps of the aggregation window.
    *   `num_snapshots`: Number of data snapshots within the window.

*   **Moment-Based Summaries (for all `FEATURES_ALL`):**
    For each feature (e.g., `cf_bid_volume_1`, `cf_imbalance_1`, `cf_bid_price_range_50`, `tb_vwap_10`, `opt_implied_volatility`, etc.):
    *   `[feature_name]_mean`: Average value of the feature.
    *   `[feature_name]_std`: Standard deviation.
    *   `[feature_name]_min`: Minimum value.
    *   `[feature_name]_max`: Maximum value.
    *   `[feature_name]_skew`: Skewness (measure of asymmetry).
    *   `[feature_name]_kurtosis`: Kurtosis (measure of "tailedness").

*   **Quantiles and Tail Ratios (for all `FEATURES_ALL`):**
    For each feature:
    *   `[feature_name]_q10`, `[feature_name]_q25`, `[feature_name]_q50`, `[feature_name]_q75`, `[feature_name]_q90`: Specific quantiles (10th, 25th, median, 75th, 90th percentile).
    *   `[feature_name]_upper_tail_ratio`: Ratio of q90 to q50.
    *   `[feature_name]_lower_tail_ratio`: Ratio of q10 to q50.
    *   `[feature_name]_tail_spread`: Difference between q90 and q10.

*   **Cross-Feature Correlations (for `CORR_FEATURES`):**
    *   `corr_[feature_i]_vs_[feature_j]`: Pearson correlation coefficient between pairs of core features.

*   **In-Window Time-Structure Metrics (for `TIME_FEATURES`):**
    For selected features:
    *   `[feature_name]_trend_slope`: Slope of a linear regression of the feature against time within the window.
    *   `[feature_name]_trend_r2`: R-squared of the linear regression.
    *   `[feature_name]_acf1`: Lag-1 autocorrelation.
    *   `[feature_name]_max_norm_jump`: Maximum normalized single-step jump.

*   **Aggregated Anomaly-Detector Statistics (for `DETECTORS`):**
    For each configured anomaly detector (e.g., `pyod_iforest`, `changepoint`, `hmm_detect`, `matrix_profile`, `level_shift`, `order_skew`):
    *   `[detector_name]_score_mean`: Average raw anomaly score within the window.
    *   `[detector_name]_score_std`: Standard deviation of raw anomaly scores.
    *   `[detector_name]_score_max`: Maximum raw anomaly score.
    *   `[detector_name]_frac_flagged`: Fraction of snapshots flagged as anomalous by the detector.
    *   `[detector_name]_first_flag_sec_from_window_start`: Time in seconds from the window start to the first anomaly flag.
    *   `[detector_name]_last_flag_sec_from_window_start`: Time in seconds from the window start to the last anomaly flag.

These detailed statistics enable quantitative analysis and comparison between pump and control events, allowing for a deeper understanding of market dynamics and anomaly characteristics.

## Troubleshooting

*   **"No data loaded from Google Drive" / Empty DataFrames:**
    *   Verify your `gdrive_api_key` and `gdrive_orderbook_root_id` / `gdrive_tradebook_root_id` in `config/pipeline_config.json` or `config/pipeline_config.yaml`.
    *   Check the exact folder structure and file names on Google Drive. The system expects `SYMBOL/YEAR/MONTH/YYYY-MM-DD.csv.gz` for orderbook and `SYMBOL/YYYY_MM/YYYY_MM_DD.csv.gz` for tradebook. Casing and delimiters are important.
    *   Review the `debug.log` file for detailed messages from `_load_gdrive_data` and `get_file_from_drive` to pinpoint where the data loading might be failing.

*   **Missing "Advanced Math" in `stats.jsonl`:**
    *   Ensure `enable_event_aggregation` is set to `true` in your backtest configuration.
    *   Verify that `features_to_aggregate` and `detectors_to_aggregate` in `BacktestConfig` are correctly populated if you wish to filter which features/detectors are included in the aggregated output. The default behavior is to include all.
    *   Check `logs/backtest_data.log` for logs related to feature extraction and anomaly detection to confirm data is being generated before aggregation.