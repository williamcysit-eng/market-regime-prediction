import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy import stats
import warnings
from itertools import product
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

end_date = datetime.now() - timedelta(days=1)
def fetch_data(risky_ticker='^GSPC', start='1927-12-30', end=end_date):
    """Fetch market data with proper risk-free rate handling"""
    risky_data = yf.download(risky_ticker, start=start, end=end, progress=False, auto_adjust=False)
    df = pd.DataFrame(index=risky_data.index)
    
    if isinstance(risky_data.columns, pd.MultiIndex):
        df['Risky_Close'] = risky_data['Adj Close'][risky_ticker] if 'Adj Close' in risky_data else risky_data['Close'][risky_ticker]
    else:
        df['Risky_Close'] = risky_data['Adj Close'] if 'Adj Close' in risky_data else risky_data['Close']
    
    df['Risky_Log_Ret'] = np.log(df['Risky_Close'] / df['Risky_Close'].shift(1))
    
    try:
        safe_data = yf.download('^IRX', start=start, end=end, progress=False)
        if not safe_data.empty:
            if isinstance(safe_data.columns, pd.MultiIndex):
                safe_yield = safe_data['Close'].iloc[:, 0]
            else:
                safe_yield = safe_data['Close']
            daily_log_rf = np.log(1 + safe_yield / 100) / 252
            df['Safe_Log_Ret'] = daily_log_rf.reindex(df.index, method='ffill').bfill()
        else:
            raise ValueError("No IRX data")
    except:
        df['Safe_Log_Ret'] = np.log(1.04) / 252
    
    df['MA_200'] = df['Risky_Close'].rolling(window=200, min_periods=200).mean()
    df['MA_50'] = df['Risky_Close'].rolling(window=50, min_periods=50).mean()
    df['Volatility'] = df['Risky_Log_Ret'].rolling(window=21, min_periods=21).std()
    df['Return_Momentum'] = df['Risky_Log_Ret'].rolling(window=21, min_periods=21).mean()
    
    df.dropna(subset=['Risky_Log_Ret', 'Volatility', 'Return_Momentum', 'MA_200'], inplace=True)
    return df

### PARAMETER OPTIMIZATION ACROSS MULTIPLE REGIMES

def generate_validation_windows(df, n_windows=5, min_train=252*5, min_val=252*2):

    total_days = len(df)
    window_size = (total_days - min_train) // n_windows
    
    splits = []
    for i in range(n_windows):
        # Expanding window approach
        train_end = min_train + (i * window_size)
        val_end = train_end + min_val
        
        if val_end >= total_days:
            break
            
        train_data = df.iloc[:train_end]
        val_data = df.iloc[train_end:val_end]
        
        splits.append({
            'train_start': df.index[0],
            'train_end': df.index[train_end],
            'val_start': df.index[train_end],
            'val_end': df.index[min(val_end, len(df)-1)],
            'train_data': train_data,
            'val_data': val_data,
            'period_name': f"Split_{i+1}_{df.index[train_end].year}-{df.index[min(val_end, len(df)-1)].year}"
        })
    
    return splits

def backtest_strategy(df, threshold=0.5, smoothing_window=15, transaction_cost=0.001): # Baseline test
    """Vectorized backtest"""
    df = df.copy()
    df['Prob_Bear_Smooth'] = df['Prob_Bear'].rolling(window=smoothing_window).mean()
    condition_sell = (
    (df['Prob_Bear_Smooth'] > threshold) & 
    (df['Risky_Close'] < df['MA_50']) |
    (df['Risky_Close'] < df['MA_200'] * 0.9)
    )
    df['Signal'] = np.where(condition_sell, 0, 1)
    df['Position'] = df['Signal'].shift(1).fillna(1)
    df['Trade'] = (df['Position'] != df['Position'].shift(1)).astype(float)
    df['Strategy_Return'] = np.where(
        df['Position'] == 1, df['Risky_Log_Ret'], df['Safe_Log_Ret']
    ) - (df['Trade'] * transaction_cost)
    return df

def calculate_metrics(returns):
    """Sharpe, Sortino, MaxDD, Calmar"""
    if len(returns) < 10 or returns.std() == 0:
        return {'sharpe': -99, 'sortino': -99, 'max_dd': 0, 'calmar': -99, 'total_ret': -1}
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else 0
    
    # Max Drawdown
    cumret = np.exp(returns.cumsum())
    running_max = np.maximum.accumulate(cumret)
    max_dd = ((cumret - running_max) / running_max).min()
    
    # Calmar
    total_ret = np.exp(returns.sum()) - 1
    calmar = (returns.mean() * 252) / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'total_ret': total_ret
    }

def cross_regime_optimization(df, splits, transaction_cost=0.001):
    
    # Parameter grid
    thresholds = np.arange(0.30, 0.81, 0.05)
    windows = [5, 10, 15, 20, 30]
    
    records = []
    
    print(f"Testing {len(thresholds)*len(windows)} parameter combos across {len(splits)} validation windows...")
    
    for thresh, win in product(thresholds, windows):
        window_scores = []
        
        for split in splits:
            # Fit GMM on this split's training data
            train_df = split['train_data']
            val_df = split['val_data'].copy()
            
            # Quick regime fit for this window
            X_train = train_df[['Return_Momentum', 'Volatility']].values
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(X_train)
            
            X_val = val_df[['Return_Momentum', 'Volatility']].values
            probs = gmm.predict_proba(X_val)
            regimes = gmm.predict(X_val)
            
            # Identify bear regime
            val_df_temp = val_df.copy()
            val_df_temp['Regime'] = regimes
            vol_by_regime = val_df_temp.groupby('Regime')['Volatility'].mean()
            bear_id = vol_by_regime.idxmax()
            val_df['Prob_Bear'] = probs[:, bear_id]
            
            # Backtest
            result = backtest_strategy(val_df, thresh, win, transaction_cost)
            metrics = calculate_metrics(result['Strategy_Return'])
            
            window_scores.append({
                'sharpe': metrics['sharpe'],
                'calmar': metrics['calmar'],
                'max_dd': metrics['max_dd'],
                'period': split['period_name']
            })
        
        # Aggregate across windows (KEY: We care about consistency, not just average)
        sharpes = [s['sharpe'] for s in window_scores]
        calmars = [s['calmar'] for s in window_scores]
        
        record = {
            'threshold': thresh,
            'window': win,
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),  # Lower is better (consistency)
            'min_sharpe': np.min(sharpes),  # Worst case scenario
            'mean_calmar': np.mean(calmars),
            'worst_dd': min([s['max_dd'] for s in window_scores]),
            'win_rate': sum([s > 0 for s in sharpes]) / len(sharpes),  # % of windows with positive Sharpe
            'sharpe_stability': np.mean(sharpes) / (np.std(sharpes) + 0.01),
            'all_scores': window_scores
        }
        records.append(record)
    
    results_df = pd.DataFrame(records)
    
    # SELECTION CRITERIA
    # 1. Must have positive Sharpe in at least 60% of windows (not just lucky once)
    # 2. Min Sharpe > 0 (prevention of major loss)
    # 3. Maximize stability (mean/std ratio) rather than raw mean
    
    candidates = results_df[
        (results_df['win_rate'] >= 0.6) &      # Works in majority of regimes
        (results_df['min_sharpe'] > -0.5) &    # Never terrible
        (results_df['mean_sharpe'] > 0.3)      # Decent average performance
    ].copy()
    
    if len(candidates) == 0:
        print("No optimal parameters found! Relaxing constraints...")
        candidates = results_df[results_df['mean_sharpe'] > 0].copy()
    
    if len(candidates) > 0:
        # Selection
        best = candidates.loc[candidates['sharpe_stability'].idxmax()]
        
        print(f"OPTIMAL PARAMETERS SELECTED:")
        print(f"   Threshold: {best['threshold']:.2f}")
        print(f"   Smoothing: {best['window']:.0f} days")
        print(f"   Avg Sharpe: {best['mean_sharpe']:.2f} (±{best['std_sharpe']:.2f})")
        print(f"   Worst Sharpe: {best['min_sharpe']:.2f}")
        print(f"   Win Rate: {best['win_rate']*100:.0f}% of validation windows")
        print(f"   Stability Ratio: {best['sharpe_stability']:.2f}")
    else:
        best = results_df.loc[results_df['mean_sharpe'].idxmax()]
        print("Using best average Sharpe (no optimal solution found)")
    
    return best, results_df

def regime_conditional_parameters(df, splits, transaction_cost=0.001):
    
    # Classify each validation window by its volatility regime
    window_regimes = []
    for split in splits:
        val_vol = split['val_data']['Volatility'].mean()
        window_regimes.append({
            'split': split,
            'avg_vol': val_vol,
            'regime': 'High_Vol' if val_vol > 0.015 else 'Low_Vol'  # 15% annualized threshold
        })
    
    # Optimize separately for each regime type
    high_vol_splits = [w['split'] for w in window_regimes if w['regime'] == 'High_Vol']
    low_vol_splits = [w['split'] for w in window_regimes if w['regime'] == 'Low_Vol']
    
    params = {}
    
    if len(high_vol_splits) >= 2:
        print(f"\nOptimizing for HIGH volatility regimes ({len(high_vol_splits)} windows)...")
        params['high'], _ = cross_regime_optimization(df, high_vol_splits, transaction_cost)
    
    if len(low_vol_splits) >= 2:
        print(f"\nOptimizing for LOW volatility regimes ({len(low_vol_splits)} windows)...")
        params['low'], _ = cross_regime_optimization(df, low_vol_splits, transaction_cost)
    
    return params, window_regimes

# EXECUTION

if __name__ == "__main__": # nest inside __name__ block to prevent random errors
    # Load data
    df = fetch_data('^GSPC', start='1927-12-30', end='2026-01-27')
    print(f"Data loaded: {df.index[0].date()} to {df.index[-1].date()}\n")
    
    # Generate multiple validation splits covering different regimes
    splits = generate_validation_windows(df, n_windows=6, min_train=252*8, min_val=252*2)
    
    print("Validation Windows Generated:")
    for s in splits:
        print(f"  {s['period_name']}: Train to {s['train_end'].date()}, Val {s['val_start'].date()} to {s['val_end'].date()}")
    
    # Method 1: Optimal Parameter Selection (single set that works across all)
    print("METHOD 1: OPTIMAL PARAMETER OPTIMIZATION")
    
    optim_params, all_results = cross_regime_optimization(df, splits, transaction_cost=0.001)
    
    # Method 2: Regime-Conditional Parameters (adaptive)
    print("METHOD 2: REGIME-CONDITIONAL OPTIMIZATION")
    
    regime_params, regime_info = regime_conditional_parameters(df, splits, transaction_cost=0.001)
    
    # Now test on true hold-out period (2020-2026 or last 20% of data)
    test_start = pd.Timestamp('2020-01-01')
    test_data = df.loc[test_start:].copy()
    
    if len(test_data) == 0:
        # If data ends before 2020, use last 20%
        test_start = df.index[int(len(df)*0.8)]
        test_data = df.loc[test_start:].copy()
    
    print(f"TESTING ON HOLD-OUT PERIOD: {test_start.date()} to {df.index[-1].date()}")
    
    # Fit final GMM on all data before test period
    train_for_test = df.loc[:test_start].copy()
    X_train = train_for_test[['Return_Momentum', 'Volatility']].values
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X_train)
    
    # Identify bear regime from training data
    train_regimes = gmm.predict(X_train)
    train_for_test_temp = train_for_test.copy()
    train_for_test_temp['Regime'] = train_regimes
    vol_by_regime = train_for_test_temp.groupby('Regime')['Volatility'].mean()
    bear_id = vol_by_regime.idxmax()
    print(f"Identified Bear Regime: {bear_id} (Vol: {vol_by_regime[bear_id]:.4f})")
    
    # Apply to test data
    X_test = test_data[['Return_Momentum', 'Volatility']].values
    probs = gmm.predict_proba(X_test)
    test_data['Prob_Bear'] = probs[:, bear_id]
    
    # Strategy A: Optimal Parameters (single setting)
    result_optim = backtest_strategy(
        test_data.copy(), 
        optim_params['threshold'], 
        optim_params['window'], 
        0.001
    )
    metrics_optim = calculate_metrics(result_optim['Strategy_Return'])
    
    # Strategy B: Regime-Conditional (uses current vol to pick parameters)
    if 'high' in regime_params and 'low' in regime_params:
        # Dynamic switching based on trailing volatility
        test_data['Regime_Vol'] = test_data['Volatility'].rolling(63).mean()
        
        # Pre-calculate both strategies
        result_high = backtest_strategy(
            test_data.copy(), 
            regime_params['high']['threshold'], 
            regime_params['high']['window'], 
            0.001
        )
        result_low = backtest_strategy(
            test_data.copy(), 
            regime_params['low']['threshold'], 
            regime_params['low']['window'], 
            0.001
        )
        
        # Combine based on volatility regime
        high_vol_mask = test_data['Regime_Vol'] > 0.015
        result_regime = test_data.copy()
        result_regime['Signal'] = np.where(high_vol_mask, result_high['Signal'], result_low['Signal'])
        result_regime['Position'] = result_regime['Signal'].shift(1).fillna(1)
        result_regime['Trade'] = (result_regime['Position'] != result_regime['Position'].shift(1)).astype(float)
        result_regime['Strategy_Return'] = np.where(
            result_regime['Position'] == 1, 
            result_regime['Risky_Log_Ret'], 
            result_regime['Safe_Log_Ret']
        ) - (result_regime['Trade'] * 0.001)
        
        metrics_regime = calculate_metrics(result_regime['Strategy_Return'])
    else:
        metrics_regime = None
    
    # Buy & Hold baseline
    metrics_bh = calculate_metrics(test_data['Risky_Log_Ret'])
    
    # Print comparison
    print(f"\n{'Strategy':<20} | {'Sharpe':<8} | {'Max DD':<10} | {'Total Ret':<10}")
    print("-" * 60)
    print(f"{'Buy & Hold':<20} | {metrics_bh['sharpe']:>7.2f} | {metrics_bh['max_dd']*100:>8.1f}% | {metrics_bh['total_ret']*100:>8.1f}%")
    print(f"{'Optimal Parameters':<20} | {metrics_optim['sharpe']:>7.2f} | {metrics_optim['max_dd']*100:>8.1f}% | {metrics_optim['total_ret']*100:>8.1f}%")
    if metrics_regime:
        print(f"{'Regime-Conditional':<20} | {metrics_regime['sharpe']:>7.2f} | {metrics_regime['max_dd']*100:>8.1f}% | {metrics_regime['total_ret']*100:>8.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Equity curves
    ax = axes[0]
    cum_bh = np.exp(test_data['Risky_Log_Ret'].cumsum())
    cum_optim = np.exp(result_optim['Strategy_Return'].cumsum())
    
    ax.plot(cum_bh.index, cum_bh, label='Buy & Hold', color='gray', alpha=0.7)
    ax.plot(cum_optim.index, cum_optim, label='Robust Strategy', color='blue', linewidth=1.5)
    if metrics_regime:
        cum_regime = np.exp(result_regime['Strategy_Return'].cumsum())
        ax.plot(cum_regime.index, cum_regime, label='Regime-Conditional', color='green', linewidth=1.5)
    
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Out-of-Sample Performance (2020-2026)')
    ax.grid(True, alpha=0.3)
    
    # Parameter stability visualization
    ax = axes[1]
    pivot = all_results.pivot_table(
        values='mean_sharpe', 
        index='window', 
        columns='threshold'
    )
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Average Sharpe Across All Validation Windows')
    
    plt.tight_layout()
    plt.savefig('optimal_optimization_results.png', dpi=150)
    plt.show()
