import tkinter as tk
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_stock_price(stock_symbol, status_label, history_label, current_price_label, future_price_label, percentage_increase_label, results_table):
    status_label.config(text="Fetching data...")
    # Get historical stock data
    stock_data = yf.Ticker(stock_symbol).history(period="1y")
    history_label.config(text=f"Historical data from {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Prepare the data for training
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data['Close'].values
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make a prediction for the next year
    next_year = len(stock_data) + 1
    future_price = model.predict([[next_year]])[0]
    current_price = stock_data['Close'][-1]
    percentage_increase = ((future_price - current_price) / current_price) * 100
    
    status_label.config(text="Prediction complete")
    
    # Display current price, future price, and percentage increase
    current_price_label.config(text=f"Current Price: {current_price:.2f}")
    future_price_label.config(text=f"Future Price in 1 Year: {future_price:.2f}")
    percentage_increase_label.config(text=f"Percentage Increase: {percentage_increase:.2f}%")
    
    # Add result to the table
    results_table.append({
        "Stock Symbol": stock_symbol,
        "Current Price": current_price,
        "Future Price": future_price,
        "Percentage Increase": percentage_increase
    })
    
    # Update the table view
    update_results_table(results_table)
    
    # Return historical stock data and predicted stock price
    return stock_data, future_price

def update_results_table(results_table):
    # Update the table view
    df = pd.DataFrame(results_table)
    df = df.set_index('Stock Symbol')
    table_df.config(text=df.to_string())

def on_predict_button_click(status_label, history_label, current_price_label, future_price_label, percentage_increase_label, results_table):
    stock_symbol = stock_entry.get().upper()
    stock_data, future_price = predict_stock_price(stock_symbol, status_label, history_label, current_price_label, future_price_label, percentage_increase_label, results_table)
    
    # Plot the graph
    plot_stock_price(stock_data, future_price, stock_symbol)

    # Save the results to a CSV file
    df = pd.DataFrame(results_table)
    df.to_csv('stock_predictions.csv')

# Create the Tkinter window
window = tk.Tk()
window.title("Stock AI")

# Create a label for entering the stock symbol
entry_label = tk.Label(window, text="Enter Stock Symbol:")
entry_label.pack(pady=5)

# Create an entry field for entering the stock symbol
stock_entry = tk.Entry(window, width=10)
stock_entry.pack(pady=5)

# Create a status label
status_label = tk.Label(window, text="")
status_label.pack(pady=5)

# Create a label for displaying historical data range
history_label = tk.Label(window, text="")
history_label.pack(pady=5)

# Create labels for displaying current price, future price, and percentage increase
current_price_label = tk.Label(window, text="")
current_price_label.pack(pady=5)

future_price_label = tk.Label(window, text="")
future_price_label.pack(pady=5)

percentage_increase_label = tk.Label(window, text="")
percentage_increase_label.pack(pady=5)

# Create a results table
results_table = []

# Create a label for displaying the results table
table_label = tk.Label(window, text="Search Results:")
table_label.pack(pady=5)

table_df = tk.Label(window, text="")
table_df.pack(pady=5)

# Create a button to fetch the stock data and make a prediction
predict_button = tk.Button(window, text="Predict Next Year Price", command=lambda: on_predict_button_click(status_label, history_label, current_price_label, future_price_label, percentage_increase_label, results_table))
predict_button.pack(pady=5)

# Start the Tkinter event loop
window.mainloop()
