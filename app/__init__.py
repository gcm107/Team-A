# flask app initialization

from flask import Flask

app = Flask(__name__)

# import routes after app creation to avoid circular imports
from app import routes

