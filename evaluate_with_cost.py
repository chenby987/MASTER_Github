import pickle
import numpy as np
import pandas as pd
import torch
import sys

sys.path.append('qlib-update')
from master import MASTERModel

def evaluate_universe(universe, prefix, d_feat, beta, top_k, fee_rate=0.0015):
    print(f"==================================================")
    print(f"Evaluating {universe} (Top {top_k} Portfolio with {fee_rate*100:.2f}% fee)")
    
    dl_test = pickle.load(open(f'data/opensource/{universe}_dl_test.pkl', 'rb'))
    
    model = MASTERModel(
        d_feat = d_feat, d_model = 256, t_nhead = 4, s_nhead = 2, 
        T_dropout_rate=0.5, S_dropout_rate=0.5,
        beta=beta, gate_input_end_index=221, gate_input_start_index=158,
        n_epochs=0, lr = 0.0, GPU = 0, seed = 0, train_stop_loss_thred = 0.0,
        save_path='model/', save_prefix=f'{universe}_{prefix}'
    )
    
    model.load_param(f'model/{universe}_{prefix}_0.pkl')
    model.fitted = True
    
    predictions, metrics = model.predict(dl_test)
    
    test_loader = model._init_data_loader(dl_test, shuffle=False, drop_last=False)
    labels = []
    for data in test_loader:
        data = torch.squeeze(data, dim=0)
        label = data[:, -1, -1].cpu().numpy()
        labels.append(label)
        
    labels = np.concatenate(labels)
    pred_df = pd.DataFrame({'score': predictions.values, 'label': labels}, index=predictions.index)
    
    daily_excess_returns = []
    daily_raw_returns = []
    
    prev_portfolio = set()
    
    for date, group in pred_df.groupby(level='datetime'):
        market_ret = np.nanmean(group['label'].values)
        
        group = group.sort_values(by='score', ascending=False)
        top_group = group.head(top_k)
        
        # Current portfolio stocks
        curr_portfolio = set(top_group.index.get_level_values('instrument'))
        
        # Calculate Turnover
        if len(prev_portfolio) == 0:
            # First day: buy all
            traded_weight = 1.0
        else:
            # Stocks sold (in prev but not in curr)
            sold = prev_portfolio - curr_portfolio
            # Stocks bought (in curr but not in prev)
            bought = curr_portfolio - prev_portfolio
            # Weight per stock is 1/top_k
            traded_weight = (len(sold) + len(bought)) / top_k
            
        fee = traded_weight * fee_rate
        
        top_ret = np.nanmean(top_group['label'].values)
        
        if not np.isnan(top_ret) and not np.isnan(market_ret):
            net_ret = top_ret - fee
            excess_ret = net_ret - market_ret
            daily_excess_returns.append(excess_ret)
            daily_raw_returns.append(net_ret)
            
        prev_portfolio = curr_portfolio

    # Excess metrics (as usually reported against benchmark)
    excess_ret_arr = np.array(daily_excess_returns)
    ar_excess = np.mean(excess_ret_arr) * 252
    vol_excess = np.std(excess_ret_arr) * np.sqrt(252)
    ir_excess = ar_excess / vol_excess if vol_excess != 0 else np.nan

    print(f"Original metrics: IC: {metrics['IC']:.4f}, ICIR: {metrics['ICIR']:.4f}, RankIC: {metrics['RIC']:.4f}, RankICIR: {metrics['RICIR']:.4f}")
    print(f"Top {top_k} Portfolio (Excess over market, with fee) - AR: {ar_excess:.4f}, IR: {ir_excess:.4f}")
    return metrics, ar_excess, ir_excess

if __name__ == '__main__':
    # CSI300 Evaluation
    evaluate_universe(universe='csi300', prefix='opensource', d_feat=158, beta=5, top_k=30, fee_rate=0.0015)
    # Nasdaq 100 Evaluation
    evaluate_universe(universe='ndx100', prefix='opensource', d_feat=158, beta=5, top_k=30, fee_rate=0.0001)
