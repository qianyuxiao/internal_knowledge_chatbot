import uuid
import numpy as np
import json
from utils.matching_engine_utils import MatchingEngineUtils


def create_index_and_endpoint(
    CHATBOT_ID: str,
    PROJECT_ID: str = "ecg-ai-416210",
    REGION: str = "europe-west1" ,
    ME_DIMENSIONS: int = 768
    )->dict:

    ME_INDEX_NAME = f"{CHATBOT_ID}-me-index"  # @param {type:"string"}
    ME_EMBEDDING_DIR = f"{CHATBOT_ID}/me-bucket"  # @param {type:"string"}


    # dummy embedding
    init_embedding = {"id": str(uuid.uuid4()), "embedding": list(np.zeros(ME_DIMENSIONS))}

    # dump embedding to a local file
    with open("embeddings_0.json", "w") as f:
        json.dump(init_embedding, f)

    # write embedding to Cloud Storage
    ! set -x && gsutil cp embeddings_0.json gs://{ME_EMBEDDING_DIR}/init_index/embeddings_0.json

    mengine = MatchingEngineUtils(PROJECT_ID, REGION, ME_INDEX_NAME)

    # Create index
    index = mengine.create_index(
        embedding_gcs_uri=f"gs://{ME_EMBEDDING_DIR}/init_index",
        dimensions=ME_DIMENSIONS,
        index_update_method="streaming",
        index_algorithm="tree-ah",
    )
    if index:
        print(index.name)

    # Deploy index to endpoint
    index_endpoint = mengine.deploy_index()
    if index_endpoint:
        print(f"Index endpoint resource name: {index_endpoint.name}")
        print(
            f"Index endpoint public domain name: {index_endpoint.public_endpoint_domain_name}"
        )
        print("Deployed indexes on the index endpoint:")
        for d in index_endpoint.deployed_indexes:
            print(f" {d.id}")

    ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()

    ME_INDEX_ID, ME_INDEX_ENDPOINT_ID

    me_dict = {
        "PROJECT_ID": PROJECT_ID,
        "LOCATION": REGION,
        "CHATBOT_NAME": CHATBOT_ID,
        "ME_INDEX_ID": ME_INDEX_ID,
        "ME_INDEX_ENDPOINT_ID": ME_INDEX_ENDPOINT_ID,
        "ME_INDEX_NAME": ME_INDEX_NAME,
        "ME_EMBEDDING_DIR": ME_EMBEDDING_DIR,
        "ME_DIMENSIONS": 768
    }

    file_path = f"{CHATBOT_ID}_me.json"

    # Open the file in write mode
    with open(file_path, "w") as json_file:
        # Dump the dictionary to the file
        json.dump(me_dict, json_file)

    print(f"Index and Endpoint created, you can find its parameters under {file_path}")

    return me_dict