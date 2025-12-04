# flask app initialization

from flask import Flask
import os

# Create flask app 
app = Flask(__name__, template_folder='templates')

# import routes after app creation to avoid circular imports
from app import routes

