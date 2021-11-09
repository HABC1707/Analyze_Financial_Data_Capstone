# Libraries
import numpy as np
from rf import return_portfolios, optimal_portfolio
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime

# Data Import Pandas Data reader from Yahoo Finance API
ticker_list = ["MSFT", "TSLA", "SPY", "JPM", "2222.SR", "AMZN", "AMX", "PLTR"]
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 10, 28)
stock_data = web.get_data_yahoo(ticker_list, start_date, end_date)
stock_data.dropna(inplace=True)

# Financial Statistics/Calculations
adj_close_prices = stock_data['Adj Close']
stock_data_daily_returns = stock_data['Adj Close'].pct_change()

# Mean rate of return%
daily_mean = stock_data_daily_returns.mean()
daily_mean.keys()
height1 = []
for key in daily_mean.keys():
    height1.append(daily_mean[key])
x_pos1 = np.arange(len(daily_mean.keys()))

# Variance
daily_var = stock_data_daily_returns.var()
daily_var.keys()
height2 = []
for key in daily_var.keys():
    height2.append(daily_var[key])
x_pos2 = np.arange(len(daily_var.keys()))

# Standard Deviation
daily_std = stock_data_daily_returns.std()
daily_var.keys()
height3 = []
for key in daily_std.keys():
    height3.append(daily_std[key])

# Correlation
print(stock_data_daily_returns.corr())

# Portfolio Optimization, Efficient Frontier and Capital Market Line
returns_quarterly = stock_data['Adj Close'].pct_change()
expected_returns = returns_quarterly.mean()
cov_quarterly = returns_quarterly.cov()
random_portfolios = return_portfolios(expected_returns, cov_quarterly)
weights, returns, risks = optimal_portfolio(returns_quarterly[1:])

# Chart Plots
# Adjusted Closing Prices
ax = adj_close_prices.plot()
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price Over Time')
plt.title('Stocks Adjusted Price')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=4)
plt.tight_layout()

# Rates of Return
ax2 = stock_data_daily_returns.plot()
plt.xlabel("Date")
plt.ylabel("ROR")
plt.title("Daily Simple Rate of Return Over time")
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=4)
plt.tight_layout()

# Individual Rates of Return
fig, axs = plt.subplots(2, 4, figsize=(35, 10))
axs[0, 0].plot(stock_data['Adj Close']['MSFT'].pct_change())
axs[0, 0].set_title("Microsoft")
axs[0, 1].plot(stock_data['Adj Close']['TSLA'].pct_change())
axs[0, 1].set_title("Tesla")
axs[0, 2].plot(stock_data['Adj Close']['SPY'].pct_change())
axs[0, 2].set_title("S&P 500 SPDR ETF")
axs[0, 3].plot(stock_data['Adj Close']['JPM'].pct_change())
axs[0, 3].set_title("J.P. Morgan")
axs[1, 0].plot(stock_data['Adj Close']['2222.SR'].pct_change())
axs[1, 0].set_title("Aramco")
axs[1, 1].plot(stock_data['Adj Close']['AMZN'].pct_change())
axs[1, 1].set_title("Amazon")
axs[1, 2].plot(stock_data['Adj Close']['AMX'].pct_change())
axs[1, 2].set_title("America Mobil")
axs[1, 3].plot(stock_data['Adj Close']['PLTR'].pct_change())
axs[1, 3].set_title("Palantir")
plt.tight_layout()

# Mean returns plot
f3 = plt.figure()
plt.bar(x_pos1, height1)
plt.xticks(x_pos1, daily_mean.keys())
plt.xlabel("Stocks")
plt.ylabel("Daily Mean")
plt.title("Daily Mean Rate of Return")
plt.tight_layout()

# Variance plot
f4 = plt.figure()
plt.bar(x_pos2, height2)
plt.xticks(x_pos2, daily_var.keys())
plt.xlabel("Stocks")
plt.ylabel("variance")
plt.title("Daily Variance")

# Standard Deviation Plot
f5 = plt.figure()
plt.bar(x_pos2, height3)
plt.xticks(x_pos2, daily_std.keys())
plt.xlabel("Stocks")
plt.ylabel("Std")
plt.title("Daily Std")

# Portfolio Optimization, Efficient Frontier and Capital Market Line
f6 = plt.figure()
random_portfolios.plot.scatter(x='Volatility', y='Returns', fontsize=12)
plt.ylabel('Expected Returns', fontsize=14)
plt.xlabel('Volatility (Std. Deviation)', fontsize=14)
plt.title('Efficient Frontier', fontsize=24)
plt.plot(risks, returns, "orange")

# Compare the set of portfolios on the EF to the
try:
    single_asset_std = np.sqrt(np.diagonal(cov_quarterly))
    plt.scatter(single_asset_std, expected_returns, marker='X', color='red', s=200)
except:
    pass

plt.show()
