from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, world!"

@app.route("/data")
def get_data():
    # Sample data for demonstration
    data = {
        "Stock": ["AAPL", "GOOGL", "MSFT"],
        "Correlation": [0.9, 0.85, 0.95]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5039,use_reloader=False)
