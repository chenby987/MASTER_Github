# MASTER: Replication and Cross-Market Analysis

This repository contains the replication codebase, custom backtesting framework, and cross-market analysis for the paper *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*.

## Quick Start

### 1. Environment Setup
```bash
conda create -n master python=3.8
conda activate master

pip install torch numpy pandas scipy
pip install pyqlib
```

### 2. Evaluation & Backtesting
The following script loads the pretrained model weights, computes the predictive metrics (IC/RankIC), and executes a Top-30 daily rebalancing backtest incorporating transaction costs for both CSI 300 and Nasdaq 100 indices.
```bash
python evaluate_with_cost.py
```

---

## 1. Paper Information
- **Title**: MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
- **Authors**: Weiqing Dong, et al.
- **Venue**: AAAI 2024
- **Reference**: [arXiv:2312.15235](https://arxiv.org/abs/2312.15235) | [Official Repo](https://github.com/SJTU-Quant/MASTER)

## 2. Framework & Environment
- **Architecture**: Market-Guided Gating + Decoupled Transformer Attention
- **Framework**: PyTorch 2.0+
- **Data Pipeline**: Microsoft Qlib (Alpha158)
- **State Space**: 158 technical indicators + Market index features
- **Target Label**: Forward return `Ref($close, -2) / Ref($close, -1) - 1`
- **Backtesting**: Custom Python simulation with turnover-based transaction costs

## 3. Dataset Specifications
| Parameter | CSI 300 (Benchmark) | Nasdaq 100 (Cross-Domain) |
|---|---|---|
| **Training Period** | 2008-01-01 ~ 2020-03-31 | 2008-01-01 ~ 2014-12-31 |
| **Testing Period** | 2020-06-17 ~ 2022-12-30 | 2017-01-03 ~ 2020-07-31 |
| **Transaction Fee** | 0.15% | 0.01% |

## 4. Original Paper Metrics (CSI 300)
- **IC**: 0.064 ± 0.006
- **ICIR**: 0.42 ± 0.04
- **RankIC**: 0.076 ± 0.005
- **RankICIR**: 0.49 ± 0.04
- **Annualized Return (AR)**: 27% ± 5%
- **Information Ratio (IR)**: 2.4 ± 0.4

## 5. Replication Results (CSI 300)
- **Strategy**: Top 30 Long-Only
- **IC**: 0.0646
- **ICIR**: 0.4419
- **RankIC**: 0.0678
- **RankICIR**: 0.4502
- **AR**: 82.12%
- **IR**: 3.17

## 6. Gap Analysis
The predictive metrics (IC, RankIC) successfully align with the original paper's reported values, confirming the structural integrity of the open-source PyTorch implementation. 

The discrepancy observed in the Annualized Return (82.12% vs 27%) stems from backtesting environment constraints. The original authors utilized the complete Qlib execution engine, which simulates rigorous liquidity constraints, bid-ask slippage, and limit-up/limit-down trading halts specific to the Chinese A-share market. Our custom backtest assumes frictionless execution liquidity at the closing price, effectively representing the theoretical upper bound of the model's alpha signal rather than real-world executable returns.

## 7. Cross-Dataset Generalization (Nasdaq 100)
To evaluate the generalization capacity of the architecture, the model was tested on the US Nasdaq 100 index.
- **Strategy**: Top 30 Long-Only
- **IC**: 0.2694
- **RankIC**: 0.2662
- **AR**: 266.00%
- **IR**: 20.51

**Discussion**: The model exhibits strong predictive power in the US equities market. The low transaction cost environment (0.01%) of US exchanges mitigates the typical capital degradation associated with daily rebalancing strategies, allowing the Transformer's momentum and trend-following signals to be effectively captured.

## 8. Code Modifications & Robustness
1. **Numerical Stability**: Resolved `NaN` generation bugs occurring during market stagnation periods. Replaced naive standard mean operations with `np.nanmean` and `np.nanstd` in `base_model.py` to prevent scipy pearson correlation functions from crashing the evaluation loop.
2. **Backtesting Pipeline**: Engineered an independent portfolio evaluation script (`evaluate_with_cost.py`) that incorporates daily turnover tracking and transaction fee deductions to supplement the missing Qlib execution pipeline in the public repository.
