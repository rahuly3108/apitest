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
    # Load and process data
    data = pd.read_csv('ind_nifty200list.csv')
    symbols_list = data['Symbol']

    # Calculate the start date as one year ago from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    all_stocks_df = pd.DataFrame()

    # Loop through each stock symbol and download the adjusted close prices
    for symbol in symbols_list:
        stock_df = yf.download(symbol + '.NS', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)['Adj Close'].reset_index()
        stock_df.columns = ['Date', symbol]
        stock_df['Date'] = stock_df['Date'].dt.date
        if all_stocks_df.empty:
            all_stocks_df = stock_df 
        else:
            all_stocks_df = pd.merge(all_stocks_df, stock_df, on='Date', how='outer')

    # Download Nifty 50 data
    nifty50_df = yf.download('^NSEI', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)['Adj Close'].reset_index()
    nifty50_df.columns = ['Date', 'Nifty50']
    nifty50_df['Date'] = nifty50_df['Date'].dt.date

    # Merge Nifty 50 data with all stocks DataFrame
    all_stocks_df = pd.merge(all_stocks_df, nifty50_df, on='Date', how='outer')

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
    agr_df = (movement_df[symbols_list] == nifty_movement.values[:, None])
    PctAgrmt = agr_df.mean() * 100
    pangan_n50 = pd.DataFrame({'Stock': PctAgrmt.index, 'Percentage Agreement with Nifty50': PctAgrmt.values})

    result_df = correlation_df.merge(variance_ratio_df, on='Stock', how='outer') \
                               .merge(reg_coeff, on='Stock', how='outer') \
                               .merge(pangan_n50, on='Stock', how='outer')
    result_df.columns = ['Stock', 'Correlation with Nifty50', 'Variance Ratio (V1/V2)', 'Regression Coefficient with Nifty50', 'Percentage Agreement with Nifty50']
    result_df = result_df.dropna()

    return result_df

@app.route("/")
def home():
    return "Hello, world!"
    
    
@app.route("/correlation")
def correlation():
    result_df = process_data()
    correlation_df = result_df[['Stock', 'Correlation with Nifty50']]
    return jsonify(correlation_df.to_dict(orient='records'))

@app.route("/regression")
def regression():
    result_df = process_data()
    reg_coeff_df = result_df[['Stock', 'Regression Coefficient with Nifty50']]
    return jsonify(reg_coeff_df.to_dict(orient='records'))

@app.route("/summary")
def summary():
    result_df = process_data()
    clustering_data = result_df[['Correlation with Nifty50', 'Variance Ratio (V1/V2)', 'Regression Coefficient with Nifty50', 'Percentage Agreement with Nifty50']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Choose the number of clusters (K)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_
    result_df['Cluster'] = cluster_labels + 1

    # Create summary with counts
    cluster_summary = result_df.groupby('Cluster')[['Correlation with Nifty50', 'Variance Ratio (V1/V2)', 'Regression Coefficient with Nifty50', 'Percentage Agreement with Nifty50']].mean().round(3)
    cluster_counts = result_df.groupby('Cluster').size()
    summary_with_count = cluster_summary.assign(Count=cluster_counts).reset_index()

    return jsonify(summary_with_count.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
