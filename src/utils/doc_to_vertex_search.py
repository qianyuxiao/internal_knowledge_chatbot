import re
from langchain_core.documents.base import Document

def split_pages_into_artiles(pages,  header_spliter = "ARTICLE"):
    articles = []
    title_ = ''
    for pn_, page in enumerate(pages):

        page_content = page.page_content
        metadata = page.metadata.copy()
        metadata['page']+=1



        for i,s_ in enumerate(page_content.split(header_spliter)):

            first_line = s_.split("\n")[0]
            len_first_line = len(first_line)


            if pn_+i==0: # if first page first split
                title_dict = {"title":s_.split("\n")[3]} #TBU
            elif i==0 & len_first_line<10: # if an article is splitted into 2 pages, combine previous content together           
                title_dict = {"title":f'{header_spliter}{title_}'}
                previous_part = articles[-1].page_content
                s_ = f"{previous_part} \n\n {s_}"
                articles = articles[:-1]
            else:
                title_ = first_line
                title_dict = {"title":f'{header_spliter}{title_}'}
                s_ = f"{header_spliter}{s_}"
            metadata_article = {**metadata,**title_dict}
            article = Document(page_content=s_, metadata=metadata_article)
            # print(metadata_article )
            articles.append(article)
    return articles

# split articles to chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_to_chunk(articles,
                  chunk_size= 1000,
                  chunk_overlap=100
                  ):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n", ".", "!", "?"] #, ".", "!", "?", ",", " ", ""
    )
    doc_splits_list = []
    for article in articles:
        doc_splits = text_splitter.split_documents([article])

        # Add chunk number to metadata
        for idx, split in enumerate(doc_splits):
            split.metadata["chunk"] = idx
        doc_splits_list+=doc_splits
    return doc_splits_list

from utils.custom_vertexai_embeddings import CustomVertexAIEmbeddings

def get_embeddings(
    EMBEDDING_QPM: int = 100,
    EMBEDDING_NUM_BATCH: int = 5
    ):
  
    embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )
    
    return embeddings

import json

def get_me_parameters(file_path):
    #!gsutil cp gs://{chatbot_id}/me_parameters/{chatbot_id}_me.json ../vector_store_me_parameters/
    #TD: to download directly from gs based on id
    # Load parameters from the JSON file into a dictionary
    with open(file_path, "r") as json_file:
        parameters = json.load(json_file)
    return parameters

from utils.matching_engine import MatchingEngine

def get_vector_store(parameters, embeddings):
    # Accessing parameters
    PROJECT_ID = parameters['PROJECT_ID']
    LOCATION = parameters['LOCATION']
    ME_INDEX_ID = parameters['ME_INDEX_ID']
    ME_INDEX_ENDPOINT_ID = parameters['ME_INDEX_ENDPOINT_ID']
    ME_EMBEDDING_DIR = parameters['ME_EMBEDDING_DIR']

    
    me = MatchingEngine.from_components(
                    project_id=PROJECT_ID,
                    region=LOCATION,
                    gcs_bucket_name=f"gs://{ME_EMBEDDING_DIR}".split("/")[2],
                    embedding=embeddings,
                    index_id=ME_INDEX_ID,
                    endpoint_id=ME_INDEX_ENDPOINT_ID,
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

