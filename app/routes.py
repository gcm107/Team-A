"""
Flask route handlers for stock prediction API endpoints.
"""

from flask import jsonify, request
from app import app
from model import predict_up_down, predict_return, get_feature_cols

# debugging hwlper
@app.route("/")
def index():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Stock Prediction API"})


@app.route("/predict/updown", methods=["POST"])
def predict_updown():
    
    # function to predict whether stock will go up (1) or down (0) in next 5 days.
    
    try:
        # get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # validate required fields
        feature_cols = get_feature_cols()
        missing_fields = [col for col in feature_cols if col not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # extract features
        daily_return = float(data["daily_return"])
        mean_5d = float(data["mean_5d"])
        vol_5d = float(data["vol_5d"])
        spy_return = float(data["spy_return"])
        
        # make prediction
        result = predict_up_down(daily_return, mean_5d, vol_5d, spy_return)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/return", methods=["POST"])
def predict_return_endpoint():
    # function to predict 5 day forward return
    try:
        # get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # validate required fields
        feature_cols = get_feature_cols()
        missing_fields = [col for col in feature_cols if col not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # extract features
        daily_return = float(data["daily_return"])
        mean_5d = float(data["mean_5d"])
        vol_5d = float(data["vol_5d"])
        spy_return = float(data["spy_return"])
        
        # make prediction
        result = predict_return(daily_return, mean_5d, vol_5d, spy_return)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

