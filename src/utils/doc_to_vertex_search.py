import re
from langchain_core.documents.base import Document
import json
from langchain_google_vertexai import (
    VertexAI,
    VertexAIEmbeddings,
    VectorSearchVectorStore
)



def get_embeddings(
    model_id = "textembedding-gecko-multilingual@001"
    ):
  
    embeddings = VertexAIEmbeddings(model_name=model_id)
    
    return embeddings



def get_me_parameters(file_path):
    #!gsutil cp gs://{chatbot_id}/me_parameters/{chatbot_id}_me.json ../vector_store_me_parameters/
    #TD: to download directly from gs based on id
    # Load parameters from the JSON file into a dictionary
    with open(file_path, "r") as json_file:
        parameters = json.load(json_file)
    return parameters



def get_vector_store(parameters, embeddings):
    # Accessing parameters
    PROJECT_ID = parameters['PROJECT_ID']
    LOCATION = parameters['LOCATION']
    ME_INDEX_ID = parameters['ME_INDEX_ID']
    ME_INDEX_ENDPOINT_ID = parameters['ME_INDEX_ENDPOINT_ID']
    ME_EMBEDDING_DIR = parameters['ME_EMBEDDING_DIR']

    
    me = VectorSearchVectorStore.from_components(
                    project_id=PROJECT_ID,
                    region=LOCATION,
                    gcs_bucket_name=f"gs://{ME_EMBEDDING_DIR}".split("/")[2],
                    embedding=embeddings,
                    index_id=ME_INDEX_ID,
                    endpoint_id=ME_INDEX_ENDPOINT_ID,
                    stream_update=True,
                    )
    return me


def add_splits_to_vector_store(doc_splits, me):
    
    texts = [doc.page_content for doc in doc_splits]
    metadatas = []

    for doc in doc_splits:
        document_name = doc.metadata.get("source", "").split("/")[-1]
        metadata = [
            {"namespace": "source", "allow_list": [doc.metadata.get("source", "")]},
            {"namespace": "document_name", "allow_list": [document_name]},
            {"namespace": "chunk", "allow_list": [str(doc.metadata.get("chunk", ""))]},
        ]
        # Check if title or page exists in metadata before adding it
        if "title" in doc.metadata:
            metadata.append({"namespace": "title", "allow_list": [str(doc.metadata["title"])]})
        if "page" in doc.metadata:
             metadata.append({"namespace": "page", "allow_list": [str(doc.metadata.get("page", ""))]})
        
        metadatas.append(metadata)

    doc_ids = me.add_texts(texts=texts, metadatas=metadatas)
    
    print(f"successfully added to vector store with {len(doc_ids)} new doc ids")
    return doc_ids

