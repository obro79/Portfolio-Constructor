import yfinance as yf
import pandas as pd
import riskfolio as rp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import plotly.express as px


def get_stock_data(tickers, start_date, end_date):
    # Clean ticker input by stripping any leading/trailing spaces
    tickers = [ticker.strip() for ticker in tickers]
    
    # Fetch data from yfinance
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Handle missing or failed downloads by dropping NaN columns
    data = data['Adj Close'].dropna(axis=1, how='all')
    
    if data.empty:
        raise ValueError("No valid data fetched. Check the tickers and date range.")
    
    return data

# Calculate daily returns and clean data
def calculate_returns(data):
    returns = data.pct_change().dropna()
    
    # Remove rows with NaN or infinite values
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure no column is entirely zero or constant (which would cause issues with covariance)
    returns = returns.loc[:, (returns != 0).any(axis=0)]
    
    # Check if returns data is empty after cleaning
    if returns.empty:
        raise ValueError("Returns data is empty or invalid after cleaning.")
    
    return returns

# Validate returns data
def validate_returns(returns):
    # Check if returns data is empty or invalid
    if returns.empty or returns.isnull().all().all():
        raise ValueError("Returns data is empty or contains only NaN values.")
    
    return True

# Calculate correlations for visualization
def calculate_correlations(returns):
    return returns.corr()

# Plot correlation heatmap with validation
def plot_correlation_heatmap(corr_matrix):
    # Check if correlation matrix is empty or contains only NaN values
    if corr_matrix.empty or corr_matrix.isnull().all().all():
        raise ValueError("Correlation matrix is empty or contains only NaN values.")
    
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    return fig

# Portfolio optimization (selectable models)
def optimize_portfolio(returns, model, risk_measure, objective):
    
    if risk_measure == 'MV (Mean-Variance)':  # Example of handling different risk measures
        risk_measure = 'MV'
    elif risk_measure == 'CVaR':
        risk = risk_measure
    elif risk_measure == 'MDD (Max Drawdown)':
        risk_measure = 'MDD'
    else:
        raise ValueError(f"Unsupported risk measure: {risk_measure}")

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='hist')
    
    # Validate the covariance matrix
    validate_covariance_matrix(port.cov)

    # Select optimization model
    if model == 'Classic':
        optimized_weights = port.optimization(model='Classic', rm=risk_measure, obj=objective)
    elif model == 'Black-Litterman':
        optimized_weights = port.optimization(model='BL', rm=risk_measure, obj=objective)
    elif model == 'Factor Model (FM)':
        optimized_weights = port.optimization(model='FM', rm=risk_measure, obj=objective)
    else:
        raise ValueError("Invalid model selected.")
    
    if optimized_weights.empty:
        raise ValueError("Optimized weights are empty.")
    
    if optimized_weights.isnull().any().any():
        raise ValueError("Optimized weights contain NaN values")
    
    return optimized_weights

# Validate the covariance matrix
def validate_covariance_matrix(cov_matrix):
    if cov_matrix.isnull().values.any() or np.isinf(cov_matrix.values).any():
        raise ValueError("Covariance matrix contains NaN or infinite values.")
    if not np.all(np.linalg.eigvals(cov_matrix.values) > 0):
        raise ValueError("Covariance matrix is not positive definite.")
    return True

# Plot portfolio composition
def plot_portfolio_pie(weights):
    # If weights is a DataFrame, extract the first column
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[:, 0]  # Assuming the first column has the weights

    # Convert weights to a DataFrame (if necessary) to ensure labels and values are properly structured
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Asset', 'Weight']  # Renaming columns for clarity

    # Create Plotly pie chart
    fig = px.pie(weights_df, values='Weight', names='Asset', 
                 title='Portfolio Composition',
                 hole=0.4,  # Creates a donut-style chart, remove this if you want a full pie
                 labels={'Weight': 'Weight', 'Asset': 'Asset'})

    # Customize the appearance of the chart
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)

    return fig

# Analyze portfolio performance (expected return, volatility)
def analyze_portfolio(returns, weights):
    portfolio_return = (returns.mean() * weights).sum()
    portfolio_volatility = (returns.cov() * weights).sum().sum() ** 0.5
    return portfolio_return, portfolio_volatility


def calculate_mean_variance_risk(returns):
    """
    Calculate the variance of portfolio returns.
    :param returns: DataFrame or array of asset returns
    :return: portfolio variance (risk)
    """
    # Assuming returns is a DataFrame of asset returns
    covariance_matrix = np.cov(returns, rowvar=False)
    portfolio_variance = np.diag(covariance_matrix).mean()  # Example: mean of diagonal elements (variances)
    return portfolio_variance

def calculate_cvar_risk(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) of the portfolio.
    :param returns: DataFrame or array of asset returns
    :param confidence_level: Confidence level for VaR (usually 0.95 for 5% quantile)
    :return: CVaR risk measure
    """
    # Sort returns to find Value at Risk (VaR)
    sorted_returns = np.sort(returns, axis=0)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_value = sorted_returns[var_index]

    # Calculate CVaR (average of the returns beyond VaR)
    cvar_value = sorted_returns[:var_index].mean()
    return cvar_value

def calculate_mdd_risk(returns):
    """
    Calculate the Maximum Drawdown (MDD) of the portfolio.
    :param returns: DataFrame or array of asset returns
    :return: Maximum drawdown value
    """
    cumulative_returns = np.cumprod(1 + returns, axis=0)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = np.min(drawdown)
    return abs(max_drawdown)  # Return the absolute value of the drawdown


def get_benchmark_data(ticker, start_date, end_date):
    """Fetch benchmark data from yfinance."""
    benchmark_data = yf.download(ticker, start=start_date, end=end_date)
    return benchmark_data['Adj Close']

def calculate_cumulative_returns(returns):
    """Calculate cumulative returns from daily returns."""
    cumulative_returns = (1 + returns).cumprod() - 1
    return cumulative_returns

def plot_portfolio_vs_benchmark_plotly(portfolio_cum_returns, benchmarks):
    """Plot portfolio cumulative returns vs benchmark cumulative returns using Plotly."""
    
    fig = go.Figure()
    
    # Plot portfolio cumulative returns
    fig.add_trace(go.Scatter(
        x=portfolio_cum_returns.index,
        y=portfolio_cum_returns,
        mode='lines',
        name='Optimized Portfolio',
        line=dict(width=2)
    ))
    
    # Plot benchmark cumulative returns
    for name, benchmark_cum_returns in benchmarks.items():
        fig.add_trace(go.Scatter(
            x=benchmark_cum_returns.index,
            y=benchmark_cum_returns,
            mode='lines',
            name=name,
            line=dict(dash='dash')  # Dashed lines for benchmarks
        ))

    # Update layout for better visualization
    fig.update_layout(
        title="Portfolio vs Benchmark Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        hovermode="x unified",
        legend=dict(x=0, y=1),
        template='plotly_white'
    )
    
    # Show the interactive plot
    return fig

# Example usage in your main function:
def compare_portfolio_to_benchmarks(returns, weights, start_date, end_date):
    # Calculate portfolio cumulative returns
    portfolio_returns = (returns * weights).sum(axis=1)
    print("Portfolio Returns:\n", portfolio_returns)
    portfolio_cum_returns = calculate_cumulative_returns(portfolio_returns)
    print("Weights:\n", weights)

    # Fetch benchmark data for NASDAQ, S&P 500, and Dow Jones
    sp500 = get_benchmark_data('^GSPC', start_date, end_date)
    nasdaq = get_benchmark_data('^IXIC', start_date, end_date)
    dow_jones = get_benchmark_data('^DJI', start_date, end_date)
    
    # Calculate benchmark cumulative returns
    sp500_returns = sp500.pct_change().dropna()
    nasdaq_returns = nasdaq.pct_change().dropna()
    dow_jones_returns = dow_jones.pct_change().dropna()
    
    sp500_cum_returns = calculate_cumulative_returns(sp500_returns)
    nasdaq_cum_returns = calculate_cumulative_returns(nasdaq_returns)
    dow_jones_cum_returns = calculate_cumulative_returns(dow_jones_returns)
    
    # Plot portfolio vs benchmarks using Plotly
    benchmarks = {
        'S&P 500': sp500_cum_returns,
        'NASDAQ': nasdaq_cum_returns,
        'Dow Jones': dow_jones_cum_returns
    }
    
    return plot_portfolio_vs_benchmark_plotly(portfolio_cum_returns, benchmarks)

def get_stock_sector(tickers):
    """Fetch the sector for each ticker."""
    sectors = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sectors[ticker] = info.get('sector', 'Unknown')  # Fallback to 'Unknown' if sector not found
        except Exception as e:
            sectors[ticker] = 'Unknown'
    return sectors

def calculate_sector_allocation(weights, sectors):
    """Aggregate portfolio weights by sector."""
    sector_weights = {}
    
    # Loop through the tickers and add the weights to their corresponding sector
    for ticker, weight in weights.items():
        sector = sectors.get(ticker, 'Unknown')  # Get the sector for the ticker
        if sector in sector_weights:
            sector_weights[sector] += weight
        else:
            sector_weights[sector] = weight
    
    # Convert the result to a Pandas Series for easy visualization
    sector_weights_series = pd.Series(sector_weights)
    
    return sector_weights_series

def plot_sector_allocation(sector_weights):
    # Create a Plotly pie chart and add the 'hole' parameter to make it a donut
    fig = px.pie(values=sector_weights, names=sector_weights.index, 
                 title="Sector Allocation", 
                 hole=0.4)  # The 'hole' parameter turns the pie chart into a donut

    # Customize the chart layout (optional)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)

    return fig

import numpy as np

# Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate the Sharpe Ratio of the portfolio."""
    excess_returns = returns.mean() - risk_free_rate
    return excess_returns / returns.std()

# Sortino Ratio
def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate the Sortino Ratio of the portfolio."""
    downside_risk = returns[returns < risk_free_rate].std()
    excess_returns = returns.mean() - risk_free_rate
    return excess_returns / downside_risk

# Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.05):
    """Calculate the Value at Risk (VaR) at a given confidence level."""
    return np.percentile(returns, 100 * confidence_level)

# Conditional Value at Risk (CVaR)
def calculate_cvar(returns, confidence_level=0.05):
    """Calculate the Conditional Value at Risk (CVaR) at a given confidence level."""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Maximum Drawdown
def calculate_max_drawdown(returns):
    """Calculate the Maximum Drawdown of the portfolio."""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def display_portfolio_risk_metrics_table(portfolio_returns):
    """Display portfolio risk metrics in a table."""

    # Calculate the risk metrics for the portfolio
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    sortino = calculate_sortino_ratio(portfolio_returns)
    var = calculate_var(portfolio_returns)
    cvar = calculate_cvar(portfolio_returns)
    max_drawdown = calculate_max_drawdown(portfolio_returns)

    # Create a DataFrame to display the metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Sharpe Ratio", "Sortino Ratio", "VaR (5%)", "CVaR (5%)", "Max Drawdown"],
        "Value": [sharpe, sortino, var, cvar, max_drawdown]
    })
    
    return metrics_df

    # Display the metrics as a table in Streamlit
    st.table(metrics_df)
    
def calculate_portfolio_returns(returns, weights):
    """Calculate the portfolio returns by multiplying returns with the asset weights."""
    # Multiply each asset's returns by its corresponding weight
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns
    
def weights_to_dataframe(weights):
    """Convert portfolio weights to a DataFrame for better display."""
    # Check if weights are a Series, otherwise convert it
    weights_df = pd.DataFrame(weights, columns=['Weights'])
    
    # Optionally sort the weights by value (if needed)
    weights_df = weights_df.sort_values(by='Weights', ascending=False)
    
    return weights_df

def weights_to_dataframe(weights):
    # If weights is a DataFrame, assume it has one column
    if isinstance(weights, pd.DataFrame):
        weights_df = weights.reset_index()
        weights_df.columns = ['Asset', 'Weight']
    
    # If weights is a Series, convert it to DataFrame
    elif isinstance(weights, pd.Series):
        weights_df = weights.reset_index()
        weights_df.columns = ['Asset', 'Weight']
    
    # If weights is a dictionary, convert it to DataFrame
    elif isinstance(weights, dict):
        weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
    
    else:
        raise TypeError("Input must be a Series, DataFrame, or dictionary.")
    
    # Sort the DataFrame by weight (optional)
    weights_df = weights_df.sort_values(by='Weight', ascending=False)
    # Apply custom styling using Pandas Styler


    return weights_df






