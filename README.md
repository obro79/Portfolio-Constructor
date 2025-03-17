# Portfolio Optimization Tool

A comprehensive web application for portfolio optimization, analysis, and visualization built with Streamlit and Riskfolio-Lib.

## Try it Live

**[Access the live application here](https://optimizeportfolio.streamlit.app/)**

## Overview

This tool allows investors and financial analysts to optimize investment portfolios using various modern portfolio theory techniques. The application provides an intuitive interface to:

- Input stock tickers and select date ranges for analysis
- Visualize correlations between assets
- Optimize portfolios using different risk models and objective functions
- Compare portfolios against market benchmarks
- Analyze sector allocations and risk metrics

## Features

- **Multiple Optimization Models**:
  - Classic Mean-Variance Optimization
  - Black-Litterman Model
  - Factor Model (FM)

- **Risk Measures**:
  - Mean-Variance (MV)
  - Conditional Value at Risk (CVaR)
  - Maximum Drawdown (MDD)

- **Objective Functions**:
  - Sharpe Ratio Maximization
  - Minimum Risk
  - Maximum Return

- **Comprehensive Visualizations**:
  - Correlation heatmaps
  - Portfolio allocation pie charts
  - Performance comparison with benchmarks
  - Sector allocation breakdowns

- **Risk Metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Maximum Drawdown

## Mathematical Formulations

### Optimization Models

#### 1. Classic Mean-Variance Optimization
The classic Markowitz portfolio optimization aims to find the optimal weights (w) by minimizing portfolio variance for a given expected return:

$$\min_{w} w^T \Sigma w$$

Subject to:
- $\sum_{i=1}^{n} w_i = 1$ (fully invested constraint)
- $w^T \mu \geq \mu_{target}$ (target return constraint)
- $w_i \geq 0$ (non-negative weights, if no short-selling allowed)

Where:
- $w$ is the vector of portfolio weights
- $\Sigma$ is the covariance matrix of returns
- $\mu$ is the vector of expected returns

#### 2. Black-Litterman Model
The Black-Litterman model incorporates investor views into the optimization by adjusting expected returns:

$$\mu_{BL} = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\mu_{prior} + P^T\Omega^{-1}Q]$$

Where:
- $\mu_{BL}$ is the Black-Litterman expected returns
- $\tau$ is a scaling parameter
- $\Sigma$ is the covariance matrix
- $P$ is the matrix that identifies assets involved in views
- $\Omega$ is the uncertainty of views
- $\mu_{prior}$ is the prior/equilibrium returns
- $Q$ is the vector of investor views

#### 3. Factor Model
The Factor Model assumes returns are driven by common factors:

$$r_i = \alpha_i + \sum_{j=1}^{k} \beta_{ij} f_j + \epsilon_i$$

Where:
- $r_i$ is the return of asset i
- $\alpha_i$ is the asset-specific return
- $\beta_{ij}$ is the sensitivity of asset i to factor j
- $f_j$ is the return of factor j
- $\epsilon_i$ is the idiosyncratic return

### Risk Measures

#### 1. Mean-Variance (MV)
Portfolio variance:
$$\sigma_p^2 = w^T \Sigma w$$

#### 2. Conditional Value at Risk (CVaR)
Expected loss given that the loss exceeds VaR:
$$CVaR_{\alpha}(X) = -\frac{1}{\alpha} \int_{0}^{\alpha} VaR_{\gamma}(X) d\gamma$$

#### 3. Maximum Drawdown (MDD)
Maximum observed loss from a peak to a trough:
$$MDD = \max_{t \in (0,T)} \left[ \max_{s \in (0,t)} X_s - X_t \right]$$

### Performance Metrics

#### 1. Sharpe Ratio
Risk-adjusted return:
$$Sharpe = \frac{R_p - R_f}{\sigma_p}$$

#### 2. Sortino Ratio
Downside risk-adjusted return:
$$Sortino = \frac{R_p - R_f}{\sigma_{down}}$$

#### 3. Value at Risk (VaR)
Maximum expected loss over a specified time period at a given confidence level:
$$P(X \leq -VaR_{\alpha}) = \alpha$$

## Data Sources

- Stock price data is fetched from Yahoo Finance via the `yfinance` package.
- Sector information is also retrieved from Yahoo Finance.

## Technologies Used

- **Streamlit**: Web application framework
- **Riskfolio-Lib**: Portfolio optimization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Visualization
- **Plotly**: Interactive visualizations
- **yfinance**: Financial data retrieval
