from flask import Flask, jsonify, request
import logging
import os
from txtai.embeddings import Embeddings
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import  cross_origin
# import json
# from security import authenticate, identity

'''This is section 4 app.py file.'''
app = Flask(__name__)
limiter = Limiter(get_remote_address,app=app, default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://")
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = 'secret'
# api = Api(app)

# --- Configuration ---
# IMPORTANT: Get index path from App Setting / Mount Path
# Example assuming Azure Files mount configured in App Settings/deployment
INDEX_PATH = os.environ.get("INDEX_MOUNT_PATH", "./mounts/index") # Default for testing
MODEL_NAME = os.environ.get("MODEL_NAME","sentence-transformers/all-mpnet-base-v2") # Must match indexing model
# --- End Configuration ---

# --- Global variable for loaded index (lazy loaded) ---
embeddings_index = None
is_loading = False # Simple flag to prevent concurrent loading attempts


def load_index_if_needed():
    """Loads the index if it hasn't been loaded for this instance yet."""
    global embeddings_index, is_loading
    # Basic check to prevent multiple loads if triggered concurrently on startup
    if embeddings_index is None and not is_loading:
        is_loading = True
        logging.info(f"Attempting to load index from: {INDEX_PATH}")
        try:
            if os.path.exists(INDEX_PATH):
                # Ensure content=True matches how index was saved/created
                temp_embeddings = Embeddings(path=MODEL_NAME, content=True)
                temp_embeddings.load(path=INDEX_PATH)
                # Check if loading actually resulted in items
                if temp_embeddings.count() > 0:
                    embeddings_index = temp_embeddings # Assign to global var
                    logging.info(f"Index loaded successfully. Count: {embeddings_index.count()}")
                else:
                    logging.error(f"Index loaded from {INDEX_PATH} but count is 0.")
                    embeddings_index = None # Ensure it remains None if loading failed
            else:
                logging.error("Index path not found: %s", INDEX_PATH)
                embeddings_index = None
        except Exception as e:
            logging.error(f"CRITICAL: Failed to load index: {e}", exc_info=True)
            embeddings_index = None
        finally:
            is_loading = False # Release lock

  
# --- HTTP Trigger Function ---
@app.route("/api/search") # Defines the route /api/search
@cross_origin(origins=["localhost:","127.0.0.1","zishenchan.com"])
@limiter.limit("100/minute")
def search_portfolio():
    query = request.args.get('query')
    limit = request.args.get('limit', default=3, type=int)

    load_index_if_needed()

    # --- Check if index loading failed ---
    if embeddings_index is None:
        logging.error("Index is not available for searching.")
        return jsonify({"error":True,"results": "Search service is unavailable or index failed to load."})
    if embeddings_index.count() == 0:
        logging.error("Index count is 0")
        return jsonify({"error":True,"results": "Search service is unavailable or index failed to load."})

    if not query:
        return jsonify({"error":True,"results": "Please pass a 'query' parameter in the query string or request body"})

    logging.info(f"Processing search for query: '{query}'")

    # --- Perform Search ---
    try:
        # Basic escaping for single quotes
        escaped_query = query.replace("'", "''")
        query_sql = f"SELECT text, answer, score FROM txtai WHERE similar('{escaped_query}') AND score >= 0.5 LIMIT {limit}"

        results = embeddings_index.search(query_sql)
        logging.info("Search successful, found %d results.", len(results))

        # Return results as JSON
        return jsonify({"error":False,"results": results})

    except Exception as e:
        logging.error(f"Error during search execution for query '{query}': {e}", exc_info=True)
        return jsonify({"error":True,"results": "Cannot execute query"})


# api.add_resource(Store, '/store/<string:name>')
# api.add_resource(Item, '/item/<string:name>')
# api.add_resource(ItemList, '/items')
# api.add_resource(StoreList, '/stores')
# api.add_resource(UserRegister, '/register')


# Name is only set to main when file is explicitly run (not on imports):
if __name__ == '__main__':
    # from db import db
    # db.init_app(app)
    app.run(port=5000, debug=True)
