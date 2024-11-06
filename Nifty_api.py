from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

def process_data():
    all_stocks_df = pd.read_csv('Stock_data.csv')

    # Select the first 10 columns and the last column
    all_stocks_df = all_stocks_df.iloc[:, :10].join(all_stocks_df.iloc[:, -1])

    # Calculate returns
    returns_df = all_stocks_df.copy()
    returns_df.set_index('Date', inplace=True)
    returns_df = returns_df.pct_change() * 100

    # Correlation and variance calculations
    correlation_matrix = returns_df.corr()
    correlation_df = correlation_matrix.iloc[:, -1].reset_index()
    correlation_df.columns = ['Stock', 'Correlation with Nifty50']

    variance_ratio_df = returns_df.var() / returns_df['Nifty50'].var()
    variance_ratio_df = pd.DataFrame(variance_ratio_df).reset_index() 
    variance_ratio_df.columns = ['Stock', 'Variance Ratio (V1/V2)'] 

    reg_coeff = (returns_df.cov()['Nifty50'] / returns_df.var()['Nifty50']).reset_index()
    reg_coeff.columns = ['Stock', 'Regression Coefficient with Nifty50']

    movement_df = returns_df.applymap(lambda x: 1 if x >= 0 else -1)
    nifty_movement = movement_df['Nifty50']

    # Ensure that only symbols in the columns are used
    symbols_list = [col for col in movement_df.columns if col != 'Nifty50']  # Remove Nifty50 if it's present in symbols_list
    valid_symbols = [symbol for symbol in symbols_list if symbol in movement_df.columns]

    agr_df = (movement_df[valid_symbols] == nifty_movement.values[:, None])
    PctAgrmt = agr_df.mean() * 100
    pangan_n50 = pd.DataFrame({'Stock': PctAgrmt.index, 'Percentage Agreement with Nifty50': PctAgrmt.values})

    result_df = correlation_df.merge(variance_ratio_df, on='Stock', how='outer') \
                               .merge(reg_coeff, on='Stock', how='outer') \
                               .merge(pangan_n50, on='Stock', how='outer')
    result_df.columns = ['Stock', 'Correlation with Nifty50', 'Variance Ratio (V1/V2)', 'Regression Coefficient with Nifty50', 'Percentage Agreement with Nifty50']
    result_df = result_df.dropna()

    # Standardize the data for clustering
    clustering_data = result_df[['Correlation with Nifty50', 'Variance Ratio (V1/V2)', 'Regression Coefficient with Nifty50', 'Percentage Agreement with Nifty50']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Choose the number of clusters (K)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_
    result_df['Cluster'] = cluster_labels + 1

    return result_df


@app.route("/")
def home():
    # Process the data and return the result as a JSON response
    result_df = process_data()

    # Convert the result_df to a dictionary and send it as JSON
    return jsonify(result_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
