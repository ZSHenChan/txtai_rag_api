from txtai.embeddings import Embeddings
data_set_version=1.2
data_set_file_name = f"dataset_v{data_set_version}.json"

import json

def process_json_dataset(filepath):
    """
    Reads a JSON file, extracts data, and transforms it into a list of dictionaries.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary has "output" and "text_input" keys.
    """
    try:
      with open(filepath, 'r') as f:
          data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    index_data = []
    pair_counter = 0
    all_ids = set()
    for category, chunks in data.items():
        print(f"Processing category: {category}")
        for item in chunks:
            chunk_id_base = item['id']
            answer_chunk = item['text']
            chunk_metadata = item['metadata']

            for i, query in enumerate(item['questions']):
                pair_id = f"{chunk_id_base}_q{i}" # Generate a unique ID for the pair
                pair_counter += 1

                # Data object stores the question, the answer, and metadata
                data_object = {
                    "text": query, # This is what txtai will embed by default if 'text' key exists
                    "answer": answer_chunk,
                    "category": category,
                    "metadata": chunk_metadata
                }

                # Append tuple: (unique_pair_id, data_object_to_index, optional_tags)
                # We use the 'question' field for embedding similarity.
                index_data.append((pair_id, data_object, None)) # Pass metadata as tags

    print(f"Prepared {len(index_data)} items for indexing.")
    return index_data



filepath = f"./mounts/data/{data_set_file_name}"

index_data = process_json_dataset(filepath)

import os

def indexing(filePath, save=False):

    # Create embeddings in dex with content enabled. The default behavior is to only store indexed vectors.
    embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": True})

    # Map question to text and store content
    embeddings.index(index_data)

    if(save):
        if not os.path.exists(index_path_qa):
            os.makedirs(index_path_qa)
        print(f"Saving QA index to {index_path_qa}...")
        embeddings.save(index_path_qa)
        print("QA Index saved.")

    try:
        # --- !!! ADD THIS CHECK !!! ---
        index_count_after = embeddings.count()
        print(f"[indexing function] Count immediately after .index() call: {index_count_after}")
        if index_count_after == 0 and len(index_data) > 0:
            print("[indexing function] CRITICAL: Index count is 0 but data was provided!")
        # --- END CHECK ---

        print("Indexing completed successfully in memory (according to try block).")

    except Exception as e:
        print(f"Error during embeddings.index(): {e}")
        return None # Return None if indexing failed

    return embeddings

index_path_qa = "./mounts/index"
embeddings = indexing(index_path_qa,save=True)
print("Program exit with no error.")
