"""
Flask application entry point for Heroku deployment.

This follows the LinkedIn Learning example structure where main.py
is the entry point for the Flask application.
"""

from app import app

if __name__ == "__main__":
    app.run(debug=True)

