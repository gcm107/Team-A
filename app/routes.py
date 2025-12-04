"""
Flask route handlers for stock prediction API endpoints.
"""

from flask import jsonify, request, render_template
from app import app
from model import predict_up_down, predict_return
from data_prep import prepare_features_from_ticker


@app.route("/")
def index():
    """Home page """
    return render_template("index.html")


@app.route("/predict/updown", methods=["POST"])
def predict_updown():
   
    # function to redict whether stock will go up (1) or down (0) in next 5 days.
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "ticker" not in data:
            return jsonify({"error": "Missing required field: 'ticker'"}), 400
        
        ticker = str(data["ticker"]).upper()
        date = data.get("date")  # optional
        
        # fetch and prepare features from ticker data
        features = prepare_features_from_ticker(ticker, date=date)
        
        # make prediction
        result = predict_up_down(
            features["daily_return"],
            features["mean_5d"],
            features["vol_5d"],
            features["spy_return"]
        )
        
        # add metadata
        result["ticker"] = ticker
        result["date"] = features["date"]
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/return", methods=["POST"])
def predict_return_endpoint():
    
    # function to predict the 5-day forward return.
    
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "ticker" not in data:
            return jsonify({"error": "Missing required field: 'ticker'"}), 400
        
        ticker = str(data["ticker"]).upper()
        date = data.get("date")  # optional
        
        # fetch and prepare features from ticker data
        features = prepare_features_from_ticker(ticker, date=date)
        
        # make prediction
        result = predict_return(
            features["daily_return"],
            features["mean_5d"],
            features["vol_5d"],
            features["spy_return"]
        )
        
        # add metadata
        result["ticker"] = ticker
        result["date"] = features["date"]
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
