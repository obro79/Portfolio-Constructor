import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from utils import get_stock_data, calculate_returns, validate_returns
from utils import optimize_portfolio, plot_portfolio_pie, analyze_portfolio, validate_covariance_matrix

### 1. Test for `get_stock_data` ###

class TestGetStockData(unittest.TestCase):

    @patch('utils.yf.download')  # Mock the yfinance download function
    def test_get_stock_data_success(self, mock_download):
        # Simulate the response from yfinance download
        mock_data = pd.DataFrame({
            'AAPL': [150, 152, 154],
            'GOOGL': [2800, 2820, 2840],
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_download.return_value = mock_data

        # Call the function
        tickers = ['AAPL', 'GOOGL']
        start_date = '2023-01-01'
        end_date = '2023-01-03'
        result = get_stock_data(tickers, start_date, end_date)

        # Check that data was downloaded and processed correctly
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 2))  # 3 rows, 2 columns

    @patch('utils.yf.download')  # Mock for failed ticker download
    def test_get_stock_data_empty(self, mock_download):
        # Simulate empty response (as if no data was found)
        mock_download.return_value = pd.DataFrame()

        tickers = ['INVALID']
        start_date = '2023-01-01'
        end_date = '2023-01-03'
        
        with self.assertRaises(ValueError) as context:
            get_stock_data(tickers, start_date, end_date)

        self.assertTrue('No valid data fetched' in str(context.exception))


### 2. Test for `calculate_returns` ###

class TestCalculateReturns(unittest.TestCase):

    def test_calculate_returns_success(self):
        # Create mock data
        data = pd.DataFrame({
            'AAPL': [150, 152, 154],
            'GOOGL': [2800, 2820, 2840],
        })

        returns = calculate_returns(data)

        # Expected values
        expected_returns = pd.DataFrame({
            'AAPL': [0.0133, 0.0132],
            'GOOGL': [0.0071, 0.0071],
        }, index=[1, 2]).round(4)  # rounding to compare

        pd.testing.assert_frame_equal(returns.round(4), expected_returns)

    def test_calculate_returns_nan(self):
        # Test with NaN data
        data = pd.DataFrame({
            'AAPL': [150, None, 154],
            'GOOGL': [2800, 2820, None],
        })

        returns = calculate_returns(data)
        self.assertEqual(returns.isnull().sum().sum(), 0)  # Ensure no NaNs


### 3. Test for `validate_returns` ###

class TestValidateReturns(unittest.TestCase):

    def test_validate_returns_success(self):
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.03, -0.02, 0.01],
        })
        result = validate_returns(returns)
        self.assertTrue(result)  # Ensure the function returns True

    def test_validate_returns_empty(self):
        returns = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            validate_returns(returns)
        self.assertTrue('Returns data is empty' in str(context.exception))


### 4. Test for `optimize_portfolio` ###

class TestOptimizePortfolio(unittest.TestCase):

    def test_optimize_portfolio_classic(self):
        # Mock data for returns
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.03, -0.02, 0.01],
        })

        # Call the function
        weights = optimize_portfolio(returns, 'Classic', 'MV', 'Sharpe')

        # Check if weights are returned properly
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(weights.sum(), 1.0)  # Weights should sum to 1

    def test_optimize_portfolio_invalid_model(self):
        # Mock data for returns
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.03, -0.02, 0.01],
        })

        # Test with an invalid model
        with self.assertRaises(ValueError) as context:
            optimize_portfolio(returns, 'InvalidModel', 'MV', 'Sharpe')

        self.assertTrue('Invalid model selected' in str(context.exception))


### 5. Test for `validate_covariance_matrix` ###

class TestValidateCovarianceMatrix(unittest.TestCase):

    def test_validate_covariance_matrix_success(self):
        cov_matrix = pd.DataFrame({
            'AAPL': [1, 0.5],
            'GOOGL': [0.5, 1],
        })
        result = validate_covariance_matrix(cov_matrix)
        self.assertTrue(result)  # Should return True if matrix is valid

    def test_validate_covariance_matrix_invalid(self):
        cov_matrix = pd.DataFrame({
            'AAPL': [1, 0.5],
            'GOOGL': [0.5, None],  # Invalid covariance matrix
        })

        with self.assertRaises(ValueError) as context:
            validate_covariance_matrix(cov_matrix)

        self.assertTrue('Covariance matrix contains NaN' in str(context.exception))


### 6. Test for `analyze_portfolio` ###

class TestAnalyzePortfolio(unittest.TestCase):

    def test_analyze_portfolio_success(self):
        # Mock returns and weights
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.03, -0.02, 0.01],
        })

        weights = pd.Series({
            'AAPL': 0.6,
            'GOOGL': 0.4,
        })

        port_return, port_volatility = analyze_portfolio(returns, weights)

        # Check return and volatility are numbers
        self.assertIsInstance(port_return, float)
        self.assertIsInstance(port_volatility, float)

    def test_analyze_portfolio_mismatch(self):
        # Mock returns and weights with shape mismatch
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.03, -0.02, 0.01],
        })

        weights = pd.Series({
            'AAPL': 0.6,  # Missing GOOGL weight
        })

        with self.assertRaises(ValueError) as context:
            analyze_portfolio(returns, weights)

        self.assertTrue('Shape mismatch between returns and weights' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
