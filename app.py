import os
import logging
from flask import Flask, jsonify, request
from txtai.embeddings import Embeddings
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import  cross_origin
# import json
# from security import authenticate, identity

# --- Configuration ---
# IMPORTANT: Get index path from App Setting / Mount Path
# Example assuming Azure Files mount configured in App Settings/deployment
INDEX_PATH = os.environ.get("INDEX_MOUNT_PATH", "./mounts/index") # Default for testing
MODEL_NAME = os.environ.get("MODEL_NAME","sentence-transformers/all-mpnet-base-v2") # Must match indexing model
# --- End Configuration ---

# --- Global variable for loaded index (lazy loaded) ---
embeddings_index = None
is_loading = False # Simple flag to prevent concurrent loading attempts

def preload_embeddings():
    """Loads the index into the global variable. Called once at startup with --preload."""
    global embeddings_index
    logging.info("Attempting to preload index...")
    logging.info(f"Using index path: {INDEX_PATH}")
    logging.info(f"Using model name: {MODEL_NAME}")

    if not os.path.isdir(INDEX_PATH): # More specific check for directory
        logging.error(f"CRITICAL: Index path directory not found during preload: {INDEX_PATH}")
        # You might want to list directory contents for debugging:
        # try:
        #     parent_dir = os.path.dirname(INDEX_PATH)
        #     logging.error(f"Contents of parent directory ({parent_dir}): {os.listdir(parent_dir)}")
        # except Exception as list_e:
        #     logging.error(f"Could not list parent directory: {list_e}")
        return # Stop if path invalid

    try:
        # Initialize and load
        temp_embeddings = Embeddings(path=MODEL_NAME, content=True, device='cpu') # Force CPU on render unless GPU instance
        temp_embeddings.load(path=INDEX_PATH)

        # Check count AFTER loading
        if temp_embeddings.count() > 0:
            embeddings_index = temp_embeddings # Assign to global variable
            logging.info(f"Index PRELOADED successfully. Count: {embeddings_index.count()}")
        else:
            logging.error(f"Index loaded from {INDEX_PATH} but count is 0 during preload.")
            # embeddings_index remains None
    except Exception as e:
        logging.error(f"CRITICAL: Failed to preload index: {e}", exc_info=True)
        # embeddings_index remains None


'''This is section 4 app.py file.'''
preload_embeddings()

app = Flask(__name__)
limiter = Limiter(get_remote_address,app=app, default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://")
  
# --- HTTP Trigger Function ---
@app.route("/api/search",methods=['GET']) # Defines the route /api/search
@cross_origin(origins=["localhost","127.0.0.1","zishenchan.com"])
@limiter.limit("100/minute")
def search_portfolio():
    query = request.args.get('query')
    limit = request.args.get('limit', default=3, type=int)

    # load_index_if_needed()

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

@app.route('/')
def home():
    # Simple health check endpoint
    if embeddings_index:
        return f"Service running. Index loaded with {embeddings_index.count()} items.", 200
    else:
        return "Service running, but index FAILED to load.", 503

# Name is only set to main when file is explicitly run (not on imports):
# if __name__ == '__main__':
#     # from db import db
#     # db.init_app(app)
#     port = os.environ.get("PORT",4000) 
#     app.run(host="0.0.0.0",port=port, debug=True)
