import logging
import json
import os
import azure.functions as func
from txtai.embeddings import Embeddings

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

SCRIPT_ROOT = os.path.dirname(__file__)
# Assumes 'index_files' folder is deployed alongside function_app.py
INDEX_PATH = os.path.join(SCRIPT_ROOT, "index_files") 
MODEL_NAME = os.environ.get("MODEL_NAME","sentence-transformers/all-mpnet-base-v2") # Must match indexing model
CONNECTION_STRING = os.environ.get("INDEX_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.environ.get("INDEX_CONTAINER_NAME")
BLOB_PREFIX = os.environ.get("INDEX_BLOB_PREFIX", "")

EMB_INDEX = None

def preload_embeddings():
    """Loads the index into the global variable. Called once at startup with --preload."""
    global EMB_INDEX
    logging.info("Attempting to preload index...")
    logging.info("Using index path (deployed with code): %s", INDEX_PATH)
    logging.info("Using model name: %s", MODEL_NAME)

    if not os.path.isdir(INDEX_PATH): # More specific check for directory
        logging.error("CRITICAL: Index path directory not found during preload: %s", INDEX_PATH)
        return func.HttpResponse("Hello. This HTTP request is received but something went wrong from out end.") 
  
    try:
        logging.info("Attempting to load index from local path: %s", INDEX_PATH)
        # Initialize and load
        EMB_INDEX = Embeddings(path=MODEL_NAME, content=True, device='cpu') # Force CPU on render unless GPU instance
        EMB_INDEX.load(path=INDEX_PATH)

        # Check count AFTER loading
        if EMB_INDEX.count() > 0:
            logging.info("Index PRELOADED successfully. Count: %s",EMB_INDEX.count())
        else:
            logging.error("Index loaded from %s but count is 0 during preload.",INDEX_PATH)
            EMB_INDEX = None
    except Exception as e:
        logging.error("CRITICAL: Failed to preload index: %s",e)
        EMB_INDEX = None

preload_embeddings()

@app.route(route="portfolio_rag_search")
def portfolio_rag_search(req: func.HttpRequest) -> func.HttpResponse:
    query = req.params.get('query')
    limit = req.params.get('limit', default=1, type=int)

    # load_index_if_needed()

    # --- Check if index loading failed ---
    if EMB_INDEX is None:
        logging.error("Index is not available for searching.")
        return func.HttpResponse(json.dumps({"error":True,"results": "Search service is unavailable or index failed to load."}),
        mimetype="application/json",status_code=400)
    if EMB_INDEX.count() == 0:
        logging.error("Index count is 0")
        return func.HttpResponse(json.dumps({"error":True,"results": "Search service is unavailable or index failed to load."}),
        mimetype="application/json",status_code=400)

    if not query:
        return func.HttpResponse(json.dumps({"error":True,"results": "Please pass a 'query' parameter in the query string or request body"}),
        mimetype="application/json",status_code=400)

    logging.info("Processing search for query: '%s'",query)

    # --- Perform Search ---
    try:
        # Basic escaping for single quotes
        escaped_query = query.replace("'", "''")
        query_sql = f"SELECT text, answer, score FROM txtai WHERE similar('{escaped_query}') AND score >= 0.5 LIMIT {limit}"

        results = EMB_INDEX.search(query_sql)
        logging.info("Search successful, found %d results.", len(results))

        # Return results as JSON
        return func.HttpResponse(json.dumps(results),
        mimetype="application/json",status_code=20)

    except Exception as e:
        logging.error(f"Error during search execution for query '{query}': {e}", exc_info=True)
        return func.HttpResponse(json.dumps({"error":True,"results": "Cannot execute query"}),
        mimetype="application/json",status_code=400)
