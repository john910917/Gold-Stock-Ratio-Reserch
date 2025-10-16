"""
使用最新爬取的數據訓練HMM模型
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# 導入現有的分析模組
import sys
sys.path.append(os.path.dirname(__file__))

def update_ratio_data():
    """更新黃金/S&P 500比率數據"""
    print("="*70)
    print("步驟 1: 更新黃金/S&P 500比率數據")
    print("="*70)
    
    try:
        # 載入S&P 500數據
        gspc = pd.read_csv('^GSPC_data.csv', index_col=0, parse_dates=True)
        print(f"[+] S&P 500數據載入成功")
        print(f"    期間: {gspc.index[0]} 到 {gspc.index[-1]}")
        print(f"    記錄數: {len(gspc)}")
        
        # 載入黃金期貨數據
        gold = pd.read_csv('黃金期貨歷史數據75~25106.csv', encoding='big5')
        date_col = gold.columns[0]
        gold[date_col] = pd.to_datetime(gold[date_col])
        gold = gold.set_index(date_col)
        gold = gold.sort_index()
        
        # 清理黃金數據
        gold_clean = gold['Close'].astype(str).str.replace('"', '').str.replace(',', '')
        gold['Close'] = pd.to_numeric(gold_clean, errors='coerce')
        gold = gold.dropna(subset=['Close'])
        
        print(f"[+] 黃金數據載入成功")
        print(f"    期間: {gold.index[0]} 到 {gold.index[-1]}")
        print(f"    記錄數: {len(gold)}")
        
        # 過濾1995年後的數據
        start_date = pd.to_datetime('1995-01-01')
        end_date = pd.to_datetime('2025-12-31')
        gspc = gspc[(gspc.index >= start_date) & (gspc.index <= end_date)]
        gold = gold[(gold.index >= start_date) & (gold.index <= end_date)]
        
        # 對齊數據
        gspc_dates = pd.to_datetime(gspc.index).date
        gold_dates = pd.to_datetime(gold.index).date
        
        gspc_df = pd.DataFrame({'close': gspc['Close']})
        gold_df = pd.DataFrame({'close': gold['Close']})
        gspc_df['date'] = pd.to_datetime(gspc_df.index).date
        gold_df['date'] = pd.to_datetime(gold_df.index).date
        
        # 合併數據
        merged = pd.merge(gspc_df, gold_df, on='date', suffixes=('_gspc', '_gold'))
        merged = merged.set_index('date')
        merged.index = pd.to_datetime(merged.index)
        
        # 計算比率
        merged['gold_sp500_ratio'] = merged['close_gold'] / merged['close_gspc']
        
        # 保存比率數據
        merged.to_csv('gold_sp500_ratio_data.csv', encoding='utf-8-sig')
        
        print(f"\n[+] 比率數據已更新 (1995-2025)")
        print(f"    共同交易日數: {len(merged)}")
        print(f"    比率期間: {merged.index[0]} 到 {merged.index[-1]}")
        print(f"    已保存到: gold_sp500_ratio_data.csv")
        
        return merged
        
    except Exception as e:
        print(f"[!] 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_hmm_model():
    """訓練HMM模型"""
    print("\n" + "="*70)
    print("步驟 2: 訓練HMM避險情緒模型")
    print("="*70)
    
    from market_risk_sentiment_hmm import (
        load_ratio_data,
        engineer_risk_sentiment_features,
        select_risk_sentiment_features,
        train_risk_sentiment_hmm,
        analyze_risk_sentiment_regimes,
        name_risk_sentiment_states
    )
    
    # 載入比率數據
    ratio_data = load_ratio_data()
    if ratio_data is None:
        print("[!] 無法載入比率數據")
        return None
    
    # 特徵工程
    df_features = engineer_risk_sentiment_features(ratio_data)
    features = select_risk_sentiment_features(df_features)
    
    # 訓練HMM模型
    model, scaler, hidden_states, state_probs = train_risk_sentiment_hmm(features, n_states=3)
    
    # 分析狀態
    sentiment_stats = analyze_risk_sentiment_regimes(df_features, hidden_states)
    sentiment_names = name_risk_sentiment_states(sentiment_stats)
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    
    with open('models/hmm_sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/sentiment_names.pkl', 'wb') as f:
        pickle.dump(sentiment_names, f)
    
    # 保存訓練信息
    training_info = {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points': len(features),
        'data_period': f"{df_features.index[0]} 到 {df_features.index[-1]}",
        'n_states': 3,
        'log_likelihood': model.score(scaler.transform(features)),
        'state_names': {k: v['name'] for k, v in sentiment_names.items()}
    }
    
    with open('models/training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)
    
    print(f"\n[+] HMM模型訓練完成")
    print(f"    訓練時間: {training_info['train_date']}")
    print(f"    數據期間: {training_info['data_period']}")
    print(f"    數據點數: {training_info['data_points']}")
    print(f"    對數似然: {training_info['log_likelihood']:.2f}")
    print(f"    狀態數量: {training_info['n_states']}")
    print(f"\n    識別的市場狀態:")
    for state, name in training_info['state_names'].items():
        print(f"      狀態{state}: {name}")
    
    # 識別當前狀態
    current_state = hidden_states[-1]
    current_probs = state_probs[-1]
    current_sentiment = sentiment_names[current_state]
    
    print(f"\n[+] 當前市場避險情緒")
    print(f"    狀態: {current_sentiment['name']}")
    print(f"    描述: {current_sentiment['description']}")
    print(f"    避險評分: {current_sentiment['sentiment_score']}/10")
    print(f"    市場情緒: {current_sentiment['market_mood']}")
    print(f"    建議黃金配置: {current_sentiment['gold_allocation']}")
    print(f"    建議股票配置: {current_sentiment['stock_allocation']}")
    
    print(f"\n    狀態概率分布:")
    for state in range(len(current_probs)):
        state_name = sentiment_names[state]['name']
        print(f"      {state_name}: {current_probs[state]*100:.2f}%")
    
    return model, scaler, sentiment_names, training_info

def train_enhanced_hmm():
    """訓練增強版HMM模型（狀態特徵分析）"""
    print("\n" + "="*70)
    print("步驟 3: 訓練增強版HMM模型（狀態特徵分析）")
    print("="*70)
    
    from hmm_enhanced_analysis import (
        load_ratio_data,
        engineer_features,
        select_features_for_hmm,
        train_hmm_model,
        analyze_state_features_detailed,
        name_states_based_on_features
    )
    
    # 載入數據
    ratio_data = load_ratio_data()
    if ratio_data is None:
        return None
    
    # 特徵工程
    df_features = engineer_features(ratio_data)
    feature_cols = ['ratio_value', 'momentum_20d', 'volatility_20d', 
                   'ratio_vs_ma50', 'trend_50d', 'rsi']
    features_for_hmm = select_features_for_hmm(df_features)
    
    # 訓練模型
    model, scaler, hidden_states, state_probs = train_hmm_model(features_for_hmm, n_states=3)
    
    # 分析狀態特徵
    state_features = analyze_state_features_detailed(df_features, hidden_states, feature_cols)
    state_names = name_states_based_on_features(state_features)
    
    # 保存模型
    with open('models/hmm_enhanced_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/enhanced_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/enhanced_state_names.pkl', 'wb') as f:
        pickle.dump(state_names, f)
    
    print(f"\n[+] 增強版HMM模型訓練完成")
    print(f"    識別的市場狀態:")
    for state, info in state_names.items():
        print(f"      狀態{state}: {info['name']}")
        print(f"        - {info['description']}")
    
    # 當前狀態
    current_state = hidden_states[-1]
    current_info = state_names[current_state]
    
    print(f"\n[+] 當前市場狀態")
    print(f"    狀態: {current_info['name']}")
    print(f"    描述: {current_info['description']}")
    print(f"    風險等級: {current_info['risk_level']}")
    
    return model, scaler, state_names

def update_crisis_warning():
    """更新危機預警分析"""
    print("\n" + "="*70)
    print("步驟 4: 更新危機預警分析")
    print("="*70)
    
    from crisis_early_warning_analysis import (
        load_data_with_volume,
        calculate_ratio_and_features,
        define_crisis_periods,
        analyze_pre_crisis_signals,
        identify_warning_signals,
        check_current_warning_signals
    )
    
    # 載入數據
    gspc, gold = load_data_with_volume()
    if gspc is None or gold is None:
        return None
    
    # 計算特徵
    df = calculate_ratio_and_features(gspc, gold)
    
    # 定義危機
    crises = define_crisis_periods()
    
    # 分析危機前信號
    pre_crisis_df = analyze_pre_crisis_signals(df, crises, lookback_days=90)
    
    # 識別預警信號
    avg_signals, warning_thresholds = identify_warning_signals(pre_crisis_df)
    
    # 檢查當前預警
    current_signals, warnings, risk_level = check_current_warning_signals(df, warning_thresholds)
    
    # 保存預警閾值
    with open('models/warning_thresholds.pkl', 'wb') as f:
        pickle.dump(warning_thresholds, f)
    
    with open('models/current_warnings.pkl', 'wb') as f:
        pickle.dump({
            'signals': current_signals,
            'warnings': warnings,
            'risk_level': risk_level,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)
    
    print(f"\n[+] 危機預警分析完成")
    print(f"    風險等級: {risk_level}")
    print(f"    預警信號數量: {len(warnings)}")
    
    if warnings:
        print(f"    活躍預警:")
        for warning in warnings:
            print(f"      {warning}")
    else:
        print(f"    [+] 當前無預警信號")
    
    return warning_thresholds, current_signals, warnings

def generate_training_report():
    """生成訓練報告"""
    print("\n" + "="*70)
    print("生成訓練報告")
    print("="*70)
    
    # 載入所有訓練信息
    with open('models/training_info.pkl', 'rb') as f:
        training_info = pickle.load(f)
    
    with open('models/sentiment_names.pkl', 'rb') as f:
        sentiment_names = pickle.load(f)
    
    with open('models/current_warnings.pkl', 'rb') as f:
        warning_info = pickle.load(f)
    
    # 生成報告
    report = f"""
{'='*70}
HMM模型訓練報告
{'='*70}

訓練時間: {training_info['train_date']}
數據期間: {training_info['data_period']}
數據點數: {training_info['data_points']}

【避險情緒HMM模型】
狀態數量: {training_info['n_states']}
對數似然: {training_info['log_likelihood']:.2f}

識別的市場狀態:
"""
    
    for state, name in training_info['state_names'].items():
        info = sentiment_names[state]
        report += f"""
  狀態{state}: {name}
    描述: {info['description']}
    避險評分: {info['sentiment_score']}/10
    市場情緒: {info['market_mood']}
    建議配置: 黃金 {info['gold_allocation']}, 股票 {info['stock_allocation']}
"""
    
    report += f"""
【危機預警系統】
風險等級: {warning_info['risk_level']}
預警信號數量: {len(warning_info['warnings'])}
更新時間: {warning_info['update_time']}
"""
    
    if warning_info['warnings']:
        report += "\n活躍預警信號:\n"
        for warning in warning_info['warnings']:
            report += f"  {warning}\n"
    else:
        report += "\n[+] 當前無預警信號，市場相對穩定\n"
    
    report += f"""
【模型文件】
- models/hmm_sentiment_model.pkl (避險情緒模型)
- models/feature_scaler.pkl (特徵標準化器)
- models/hmm_enhanced_model.pkl (增強版模型)
- models/warning_thresholds.pkl (預警閾值)
- models/training_info.pkl (訓練信息)

{'='*70}
"""
    
    # 保存報告
    with open('models/training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"[+] 訓練報告已保存到: models/training_report.txt")

def main():
    """主函數"""
    print("="*70)
    print("使用最新數據訓練HMM模型")
    print("="*70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 步驟1: 更新比率數據
    ratio_data = update_ratio_data()
    if ratio_data is None:
        print("[!] 無法更新比率數據，訓練中止")
        return
    
    # 步驟2: 訓練避險情緒HMM模型
    sentiment_model = train_hmm_model()
    if sentiment_model is None:
        print("[!] 避險情緒模型訓練失敗")
        return
    
    # 步驟3: 訓練增強版HMM模型
    enhanced_model = train_enhanced_hmm()
    
    # 步驟4: 更新危機預警
    warning_result = update_crisis_warning()
    
    # 步驟5: 生成訓練報告
    generate_training_report()
    
    print("\n" + "="*70)
    print("所有模型訓練完成！")
    print("="*70)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n模型文件已保存到 models/ 目錄")
    print("訓練報告已保存到 models/training_report.txt")
    print("="*70)

if __name__ == "__main__":
    main()
