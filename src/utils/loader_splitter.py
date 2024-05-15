from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
import os
from utils.general_utils import *
import nltk

def nb_tokens(text):
    return len(nltk.word_tokenize(text))

def get_docs_min_max_token_length(docs):
    min_nb = 1000
    min_idx = 0
    max_nb = 0
    max_idx = 0
    for _, doc in enumerate(docs):
        nb_ = nb_tokens(doc.page_content)
        if nb_>max_nb:
            max_nb = nb_
            max_idx = _
        if nb_<min_nb:
            min_nb = nb_
            min_idx = _
    return {min_idx:min_nb,max_idx:max_nb}
    
def load_pdf_to_langchain_doc(
    pdf_path,
    unstructure_loader: bool=False):
    
    if unstructure_loader:
        print("Using unstrucure loader")
        loader = UnstructuredPDFLoader(f'{pdf_path}')
    else:
        loader = PyPDFLoader(f'{pdf_path}')
    doc = loader.load()
    
    return doc


def process_folder(folder_path,
                   unstructure: bool=False
                  ):
    docs = []
    loaded_pdfs = []  
    unloaded_pdfs = []
    other_files = []
    
    for item in os.listdir(folder_path):
        
        item_path = os.path.join(folder_path, item)
        print_with_time(f"Scanning {item_path}")
        
        if os.path.isdir(item_path):
            #if it's folder go to next level
            print_with_time(f"{item_path} is folder, now go to new level")
            process_folder(item_path)
            
        elif item.lower().endswith('.pdf'):

                try:
                    
                    doc = load_pdf_to_langchain_doc(item_path,unstructure_loader=unstructure)
                    if len(doc[0].page_content)>10:
                        docs += doc
                        loaded_pdfs.append(item_path)    
                    else:
                        unloaded_pdfs.append(item_path)    
                    
                except Exception as e:
                    unloaded_pdfs.append(item_path)     
        else:
            other_files.append(item_path)
    return docs, unloaded_pdfs, loaded_pdfs, other_files

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