# Smart Supply

Minimizes perishable food waste and maximize profit using demand forecasting and machine learning.

## Features

- Dual-target prediction: both Sales Volume and Reorder Quantity
- Trains four ML models: Random Forest, Gradient Boosting, KNN, and Linear Regression
- Baseline naive model for comparison
- Proper train/validation/test splitting - 60/20/20
- Forecasts are noisy and realistic (market variance simulation)
- Calculates projected weekly revenue, profit, and inventory savings
- Ranks products by restock urgency
- Saves full prediction results to a CSV file
- Visualizes MAE and RMSE model performance

## Setup

```bash
# Clone the repo
git clone https://github.com/ToxicChunk/smart-supply.git
cd smart-supply

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
