from datetime import datetime
import time
import streamlit as st
from utils.chatbot import *
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (Namespace, NumericNamespace)

#local model packages
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import TextIteratorStreamer
from threading import Thread

def st_write_speed(start,end):
    speed = (end-start).seconds
    st.write(f'{speed}s used')
    return speed

def print_info(msg):
    print("-"*100)
    print(msg)
    

def greet_msg(department_option):
    greet_msg  = f"Greetings! I'm here to assist you with any inquiries concerning ** {department_option}**. Should you wish to explore other subjects, feel free to update your topic option on the left-hand side. "
    for word in greet_msg.split():
        yield word + " "
        time.sleep(0.03)

def init_st_session_state(vector_store_option,
                          model_option,
                          show_ref_content_option):
    print("Initiating session state....")
    st.session_state.messages = []
    st.session_state.vector_store_option = vector_store_option
    st.session_state.model_option = model_option
    st.session_state.show_ref_content_option = show_ref_content_option
    st.session_state.memory =  get_memory() 
    st.session_state.memory.clear()
    show_message()
        
def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()
  
# chatbot related cache function    
# local models
@st.cache_resource
def load_local_model(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
               ):
    # use quantization to lower GPU usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    return model,tokenizer, terminators,streamer

@st.cache_resource
def get_response(messages,
                 llm_temperature,
                 llm_top_p,
                 llm_top_k,
                 model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
                 ):
    model,tokenizer, terminators,streamer = load_local_model(model_id)
    
    # apply chat template
    messages_tmpl = tokenizer.apply_chat_template(messages,
                        tokenize=False,
                        add_generation_prompt=True
                        )

    # tokenize msgs
    inputs = tokenizer([messages_tmpl], return_tensors="pt").to('cuda')
    
    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(inputs, 
                            streamer=streamer, 
                            max_new_tokens=1500,
                            eos_token_id=terminators,
                            top_k= llm_top_k,
                            top_p= llm_top_p,
                            do_sample=True,
                            temperature=llm_temperature,
                            )
    # chat
    thread = Thread(target=model.generate,
                    kwargs=generation_kwargs)
    thread.start() 
    
    for _, new_text in enumerate(streamer):
        yield new_text
    
def format_prompt(prompt,retrieved_documents):
    """using the retrieved documents we will prompt the model to generate our responses"""
    formatted_prompt = f"Question:{prompt}\nContext: \n"
    separated_line = "-"*50+"\n"
    for idx,doc in enumerate(retrieved_documents) :
        formatted_prompt+= f"{separated_line} Ranked Document: {idx+1} \nName: {doc.metadata['document_name']}\nContent: {doc.page_content} \n"
    return formatted_prompt

def processed_msgs(refined_question,retrieved_documents):
    
    SYS_PROMPT = """You are an assistant for answering questions.
    You are given the extracted parts of several documents rank by relevance with document name and a question. Please pick the most relevant document related to the question then 
    provide a conversational answer in the same language as the question. Always cite the referred document in your response.
    If you don't know the answer, just say "I do not know." Don't make up an answer."""
    
    formatted_prompt = format_prompt(refined_question,retrieved_documents)
    
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    
    return messages
    
# Vector store 
@st.cache_resource
def load_vector_store(vector_store_option):
    print_info(f"Loading vector store...")
    vector_store = get_vector_store_by_topic(vector_store_option)  
    return vector_store


def reset_vector_store():
    load_vector_store.clear()

def get_google_doc_link(vector_store_option):
    if vector_store_option=="DPO":
        g_link = "https://drive.google.com/drive/folders/1aIxinK3LugUGlNIr23mQSTxqn8fSdxsi"
    elif vector_store_option == "HR":
        g_link = "https://drive.google.com/drive/folders/1zNhiZvSny3rpP18OGfW4O8JSHqCF9QRK"
    else:
        g_link = "https://drive.google.com/drive/folders/1Rr9F7N1HYNLqIyAcVdqtoMg6BuhhdKZq"
    return g_link

def show_message():       
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
def main():
    ## Set title and sidebar
    
    ## Header
    title = "ECG Internal Knowledge Chatbot"
    st.set_page_config(page_title="ECG Internal Knowledge Chatbot", page_icon="https://jobs.europeancampinggroup.com/generated_contents/images/company_logo_career/ZMbqNXm9-logo-ecg-new-small.png")
    st.title(title)
    
    ## Sidebar
    st.sidebar.image("https://jobs.europeancampinggroup.com/generated_contents/images/company_logo_career/ZMbqNXm9-logo-ecg-new-small.png")
    st.sidebar.title('Usage and Limitation')
    st.sidebar.write("- LLMs generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.")
    st.sidebar.markdown("- You can find the original documents that chatbot is retrieving knowledge from [here](https://drive.google.com/drive/folders/1TtxvSAeDBQh50r18OOOIoUyCbbAfHq-W)")
    
    ### Choice box
    department_option = st.sidebar.selectbox(
        'Department',
        ("dpo","insurance"))
    
    pdfloader_option = st.sidebar.selectbox(
        'PDF Loaded Method',
        ("unstructured","structured"))
    
    chunk_option = st.sidebar.selectbox(
        'Chunk Size',
        (500.0,2000.0,4000.0))
    
    vector_store_option = st.sidebar.selectbox(
        'Vector Store',
        ("Vertex"))
    
    search_distance = st.sidebar.slider('Retriver search distance',0.0, 1.0, 0.70)
    search_nb_docs = st.sidebar.slider('Retriver search maximum docs',1, 8, 4)
    llm_temperature = st.sidebar.slider('Chat model temperature',0.0, 1.0, 0.10)
    llm_top_p = st.sidebar.slider('Chat model top p',0.0, 1.0, 0.75)
    llm_top_k = st.sidebar.slider('Chat model top k',1, 40, 20)
    
    # model_option = st.sidebar.selectbox(
    #     'Chatbot model',
    #     ("chat-bison", "gemini-pro" ))
    
    model_option = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    show_ref_content_option = st.sidebar.selectbox(
        'Show Ref. Content?',
        (False, True))
    ### Buttons
    st.sidebar.button('Reset Chat', on_click=reset_conversation, type="primary")
    st.sidebar.button('Clear Vector Store Cache', on_click=reset_vector_store, type="primary")
    
    # Load and cache chatbot element
    me = load_vector_store("All")
    g_link = get_google_doc_link(vector_store_option)

    
    # Show messages
    if 'messages' not in st.session_state:
        init_st_session_state(vector_store_option,
                              model_option,
                              show_ref_content_option)
        # memory.clear()
        with st.chat_message("assistant"):
            st.write(greet_msg(department_option))
    

    show_message()
    
    # Chat flow
    question_example = "Saisissez votre question ici, par exemple : Ce qui n'est pas inclus dans le prix de la réservation ?"
    if question_prompt := st.chat_input(question_example):
        st.session_state.messages.append({"role": "user", "content": question_prompt})
        with st.chat_message("user"):
            st.markdown(question_prompt)
            
        # get answer and stream answer
        start = datetime.now()
        
        
        #get refined question based on conversation history
        if len(st.session_state.messages)>2:
            refined_question = query_refiner(st.session_state.messages,question_prompt)
            st.write(f"Refined question based on conversation history: {refined_question}")
        else:
            refined_question = question_prompt
        
        # Set vector store filter 
        filters = [Namespace(name="department", allow_tokens=[department_option]),
           Namespace(name="loader_method", allow_tokens=[pdfloader_option])
          ]


        numeric_filters = [NumericNamespace(name="chunk_size", value_float=chunk_option, op="EQUAL")]
        
        # retrive documents
        contexts = me.similarity_search(refined_question,
                                        k=search_nb_docs,
                                        search_distance=search_distance, 
                                        filter=filters, 
                                        numeric_filter=numeric_filters)
        
        # process messgaes
        messages = processed_msgs(refined_question,contexts)
        
        # get response
        response = get_response(messages,
                    llm_temperature,
                    llm_top_p,
                    llm_top_k,
                    model_id = model_option
                    )
            
        with st.chat_message("assistant"):
            st.write(response)
            print(response)
            #get reference documents
            source_and_articles = set()
            for _, s_ in enumerate(contexts):
                source = s_.metadata.get("source")
                source = '/'.join(source.split('/')[3:])
                title = s_.metadata.get("title")
                page = s_.metadata.get("page")
                page_content = s_.page_content
                print(s_)
                title_msg = ""
                if title !=None:
                    title_msg = f"Title {title}, "
                page_msg=""
                if page !=None:
                    page = int(page) + 1
                    page_msg = f"Page {page}, "            
                if show_ref_content_option:
                    source_and_articles.add(f"""{source} {page_msg}{title_msg}.\n\n Detailed content: {page_content} """)
                else:
                    source_and_articles.add(f"""{source} {page_msg}{title_msg} """)
                    
            st.write("-"*100)
            st.markdown(f"Info based on documents under [google drive]({g_link})")
            if len(contexts)>0 and response.strip()!="Do you want to ask a question?":
                # show reference documents                    
                for item in source_and_articles:
                    st.write(item)
                
        end = datetime.now()
        st_write_speed(start,end)

        st.session_state.messages.append({"role": "assistant", "content": f"{response}"})

if __name__ == "__main__":
    main()
