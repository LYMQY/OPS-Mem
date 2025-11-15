# -*- coding: utf-8 -*-
# Define constants
initial_price = 1.50
max_reduction = 0.50  # Since reducing more than $0.50 would make price negative
fixed_cost = 2000
marginal_cost = 0.50
items_per_10_cents = 1000

# Initialize variables to track maximum profit
max_profit = -float('inf')
best_price = 0
best_quantity = 0

# Try all valid prices (from $1.50 to $1.00 by steps of $0.10)
for price in range(int(150), int(100), -1):
    price_dollars = price / 100.0
    quantity_sold = 5000 + (initial_price - price_dollars) * items_per_10_cents / 0.10
    quantity_sold = int(quantity_sold)  # Round to integer (since you can't sell a fraction of an item)
    revenue = price_dollars * quantity_sold
    cost = marginal_cost * quantity_sold + fixed_cost
    profit = revenue - cost

    # Update best price and quantity if profit is higher
    if profit > max_profit:
        max_profit = profit
        best_price = price_dollars
        best_quantity = quantity_sold

# Print the results
print('-' * 10)
print(f"Best Price per Item: ${best_price:.2f}")
print(f"Number of Items Sold: {best_quantity}")
print(f"Maximum Profit: ${max_profit:.2f}")