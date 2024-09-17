import streamlit as st
import pandas as pd
from utils import get_stock_data, calculate_returns, calculate_correlations, compare_portfolio_to_benchmarks, plot_sector_allocation, get_stock_sector
from utils import plot_correlation_heatmap, optimize_portfolio, plot_portfolio_pie, analyze_portfolio, validate_returns, calculate_sector_allocation, display_portfolio_risk_metrics_table
from utils import *

def main():
    
    st.title("Portfolio Optimization Tool")

    # Step 1: Input stock tickers and date range
    tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, GOOGL")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2023-01-01"))

    # Step 2: Model selection
    model = st.sidebar.selectbox("Select Optimization Model", ["Classic", "Black-Litterman", "Factor Model (FM)"]) ##

    # Step 3: Risk measure selection
    risk_measure = st.sidebar.selectbox("Select Risk Measure", ["MV (Mean-Variance)", "CVaR", "MDD (Max Drawdown)"]) 

    # Step 4: Objective function selection
    objective = st.sidebar.selectbox("Select Objective Function", ["Sharpe", "MinRisk", "MaxRet"])

    # Fetch stock data
    if tickers:
        tickers_list = tickers.split(',')
        try:
            

            data = get_stock_data(tickers_list, start_date, end_date)

            # Calculate returns
            returns = calculate_returns(data)

            # Validate returns data before proceeding
            validate_returns(returns)

            # Correlation Matrix
            correlations = calculate_correlations(returns)
            
            # Plot only if the correlation matrix is valid
            st.pyplot(plot_correlation_heatmap(correlations))
            
            

            # Portfolio Optimization
            weights = optimize_portfolio(returns, model, risk_measure, objective)
            
            sharpe_weights = optimize_portfolio(returns, model, risk_measure, objective='Sharpe')
            minrisk_weights = optimize_portfolio(returns, model, risk_measure, objective='MinRisk')
            maxret_weights = optimize_portfolio(returns, model, risk_measure, objective='MaxRet')
            
            #st.write("Optimized Portfolio Weights", weights)
            weights_df = weights_to_dataframe(weights)
            styled_df = weights_df.style.set_properties(**{
                'font-size': '20px',       
                'width': '2000px',          
            }).format("{:.2%}", subset=['Weight'])
            
            st.table(styled_df)
            
            col1, spacer, col2, spacer2, col3 = st.columns([1, 0.2, 1, 0.2, 1])

            with col1:
                st.header("Maximized Sharpe")
                st.plotly_chart(plot_portfolio_pie(sharpe_weights))

            with col2:
                st.header("Minimum Risk")
                st.plotly_chart(plot_portfolio_pie(minrisk_weights))

            with col3:
                st.header("Maximized Sharpe")
                st.plotly_chart(plot_portfolio_pie(maxret_weights))

            
            weights = weights.squeeze()
            weights = weights.reindex(returns.columns)

            # Portfolio Analysis (Return, Volatility)
            st.plotly_chart(compare_portfolio_to_benchmarks(returns, weights, start_date, end_date))
            # Sector Allocation Visualization
            sectors = get_stock_sector(tickers_list)  # Fetch sector info for the tickers
            sector_weights = calculate_sector_allocation(weights, sectors)  # Calculate sector allocation
            st.plotly_chart(plot_sector_allocation(sector_weights))  # Plot sector allocation pie chart
            
            sharpe = calculate_sharpe_ratio(returns)
            sortino = calculate_sortino_ratio(returns)
            var = calculate_var(returns)
            cvar = calculate_cvar(returns)
            max_drawdown = calculate_max_drawdown(returns)
            
            portfolio_returns = calculate_portfolio_returns(returns, weights)

            # Display Portfolio Risk Metrics
            st.table(display_portfolio_risk_metrics_table(portfolio_returns))

        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
