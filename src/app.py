from datetime import datetime
import streamlit as st
from streamlit_feedback import streamlit_feedback
# from trubrics.integrations.streamlit import FeedbackCollector
import uuid

#feedback related
import uuid
import datetime as dt
from google.cloud import bigquery
from utils.bq_utils import load_json_to_bq
from streamlit_feedback import streamlit_feedback

def print_info(msg):
    print("-"*100)
    print(msg)
    
# vector store 
from utils.doc_to_vertex_search import *
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (Namespace,NumericNamespace)

@st.cache_resource
def get_vector_db():
    parameters = get_me_parameters('vector_store_me_parameters/ecg_all_me.json')
    embeddings = get_embeddings()
    db = get_vector_store(parameters, embeddings)
    return db

def retrieve_contexts(question,
                      department_option,
                      pdfloader_option,
                      chunk_option
                      ):
    
    db = get_vector_db()
    
    # set retriever filters
    filters = [Namespace(name="department", allow_tokens=[department_option]),
           Namespace(name="loader_method", allow_tokens=[pdfloader_option])
          ]


    numeric_filters = [NumericNamespace(name="chunk_size", value_float=chunk_option, op="EQUAL")]
    
    # retrive documents
    search_nb_docs = 4
    search_distance = 0.7
    
    contexts = db.similarity_search(question,
                                    k=search_nb_docs,
                                    search_distance=search_distance, 
                                    filter=filters, 
                                    numeric_filter=numeric_filters)
    return contexts

# question refiner
from vertexai.language_models import TextGenerationModel
def query_refiner(conversation:str, 
                  query:str,
                  text_model = "text-bison@002"):
    '''
    Refine user's query based on conversation history.
    ###
    conversation: previous conversation history
    query: last question that user posted
    output: refined query based on conversation history context
    '''
    model = TextGenerationModel.from_pretrained(text_model)
    response = model.predict(
    prompt=f"Given the following user query and conversation log, formulate a question that includes all the entities or subject mentioned would be the most relevant to provide the user with an answer from a knowledge base specifying the subject if it's mentioned.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0,
    max_output_tokens=256
    )
    return response.text

# prompt related
SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of several documents rank by relevance with document name and a question. Please pick the most relevant document related to the question then provide a conversational answer in the same language as the question. Always cite the referred document in your response.
Strictly Use ONLY the provided context to answer the question at the end. Think step-by-step as below and then answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

def format_prompt(prompt,retrieved_documents):
  """using the retrieved documents we will prompt the model to generate our responses"""
  formatted_prompt = f"Question:{prompt}\nContext: \n"
  separated_line = "-"*50+"\n"
  for idx,doc in enumerate(retrieved_documents) :
    formatted_prompt+= f"{separated_line} Ranked Document: {idx+1} \nName: {doc.metadata['document_name']}\nContent: {doc.page_content} \n"
  return formatted_prompt

# vertex chat model
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate

def get_response_vertex(
    question,
    retrieved_documents,
    model_id
    ):
    model = VertexAI(
        model_name=model_id, #"gemini-1.0-pro", #"text-bison@002",
        max_output_tokens=1500,
        temperature=0,
        verbose=True,
        streaming = True
        )
    
    formatted_prompt = SYS_PROMPT + "\n" + format_prompt(question,retrieved_documents)
    
    print_info('formatted_prompt')
    print(formatted_prompt)
    print_info("")
    
    template = """{question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | model

    return chain.invoke({"question": formatted_prompt})

# local models
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch
from threading import Thread
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
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    return model,tokenizer,streamer

def get_response_local_llm(question,
                    retrieved_documents,
                    model_id
                    ):
    model,tokenizer, streamer = load_local_model(model_id)
    
    formatted_prompt = format_prompt(question,retrieved_documents)
    
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    
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
                            pad_token_id=tokenizer.eos_token_id,
                            # eos_token_id=terminators,
                            # top_k= llm_top_k,
                            # top_p= llm_top_p,
                            do_sample=True,
                            temperature=0.0,
                            )
    # chat
    thread = Thread(target=model.generate,
                    kwargs=generation_kwargs)
    thread.start() 
    
    for _, new_text in enumerate(streamer):
        yield new_text
        
# all chat model combined
def get_response(
    question,
    retrieved_documents,
    model_id
    ):
    print_info(f"model_id: {model_id}")
    
    if model_id!="meta-llama/Meta-Llama-3-8B-Instruct":
        return get_response_vertex(
                        question,
                        retrieved_documents,
                        model_id
                        )
    else:
        return get_response_local_llm(
                        question,
                        retrieved_documents,
                        model_id
                        )
        
# st related functions
def show_message():       
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def st_write_speed(start,end):
    speed = (end-start).seconds
    st.write(f'{speed}s used')
    return speed

# feedback related
## Parameters
APP_TYPE = "rag"
FEEDBACK_OUTPUT_TABLE = f"ecg-big-data-sandbox.qianyu_test.chatbot_evaluation_{APP_TYPE}"

## st session state
def incr_run_id():
    st.session_state.run_id +=1

def update_session_state(model_id,
                         department_option,
                         pdfloader_option,
                         chunk_option,
                         question_prompt:str,
                         response: str,
                         response_time_s
                         ):
    st.session_state.model_id = model_id
    st.session_state.department_option = department_option
    st.session_state.pdfloader_option = pdfloader_option
    st.session_state.chunk_option = chunk_option
    st.session_state.question_prompt = question_prompt
    st.session_state.response = response
    st.session_state.response_time_s = response_time_s

def init_session_state():
    st.session_state.messages = []
    st.session_state.feedback = dict()
    st.session_state.uuid = str(uuid.uuid4())
    st.session_state.run_id = 0
    st.session_state.question_prompt = None
    st.session_state.response = None
    st.session_state.response_time_s = None
    st.session_state.model_id = None
    st.session_state.department_option = None
    st.session_state.pdfloader_option = None
    st.session_state.chunk_option = None

        
def get_feedback_dict(
                      score):
    feedback_dict = {
        "app_type": APP_TYPE,
        "uuid": st.session_state.uuid,
        "run_id": st.session_state.run_id,
        "feedback_time":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": st.session_state.model_id,
        "department_option":st.session_state.department_option,
        "pdfloader_option":st.session_state.pdfloader_option,
        "chunk_option":st.session_state.chunk_option,
        "question": st.session_state.question_prompt,
        "response": st.session_state.response,
        "response_time_s": st.session_state.response_time_s,
        "score": score
    }
    return feedback_dict

FEEDBACK_SCHEMA = [
    bigquery.SchemaField("uuid", "STRING"),
    bigquery.SchemaField("run_id", "INT64"),
    bigquery.SchemaField("app_type", "STRING"),
    bigquery.SchemaField("model_id", "STRING"),
    bigquery.SchemaField("department_option", "STRING"),
    bigquery.SchemaField("pdfloader_option", "STRING"),
    bigquery.SchemaField("chunk_option", "FLOAT64"),
    bigquery.SchemaField("feedback_time", "TIMESTAMP"),
    bigquery.SchemaField("response_time_s", "INT64"),
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("response", "STRING"),
    bigquery.SchemaField("score", "FLOAT64")
]

# initiate messages
if 'messages' not in st.session_state:
    init_session_state()
           
def main():
    ## Set title and sidebar
    
    ## Header
    title = "ECG Internal Knowledge Chatbot - Test Version"
    st.set_page_config(page_title="ECG Internal Knowledge Chatbot - Test Version", page_icon="https://jobs.europeancampinggroup.com/generated_contents/images/company_logo_career/ZMbqNXm9-logo-ecg-new-small.png")
    st.title(title)
    
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
    
    model_option = st.sidebar.selectbox(
        'Chat Model',
        ("gemini-1.0-pro", "meta-llama/Meta-Llama-3-8B-Instruct","text-bison@002"))
    
    chat_option = st.sidebar.selectbox(
        'In Memory Chat?',
        (False,True))
    
    show_ref_content_option = st.sidebar.selectbox(
        'Show Ref. Content?',
        (False, True))    
    
    #show message
    show_message()
    
    # chat flow
    question_example = "Saisissez votre question ici, par exemple : Ce qui n'est pas inclus dans le prix de la réservation ?"
    if question_prompt := st.chat_input(question_example):
        st.session_state.messages.append({"role": "user", "content": question_prompt})
        with st.chat_message("user"):
            st.markdown(question_prompt)
            
        # get answer and stream answer
        start = datetime.now()
        
        #get refined question based on conversation history
        if chat_option == True & len(st.session_state.messages)>2:
            refined_question = query_refiner(st.session_state.messages,question_prompt)
            st.write(f"Refined question based on conversation history: {refined_question}")
        else:
            refined_question = question_prompt
        
        # retrieve documents
        contexts = retrieve_contexts(refined_question,
                                                department_option=department_option,
                                                pdfloader_option=pdfloader_option,
                                                chunk_option=chunk_option)
        
        # get response
        response = get_response(refined_question,contexts,model_option)
        
        #st write
        with st.chat_message("assistant"):
            
            import types
            if isinstance(response, types.GeneratorType):
                response = st.write_stream(response)
            else:
                st.write(response)
            print("response:","*"*50)
            print(response)
            print("*"*50)
            
            #get reference documents
            source_and_articles = set()
            for _, s_ in enumerate(contexts):
                source = s_.metadata.get("source")
                source = '/'.join(source.split('/')[3:])
                title = s_.metadata.get("title")
                page = s_.metadata.get("page")
                page_content = s_.page_content
                #print(s_)
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
            g_link = "https://drive.google.com/drive/folders/1TtxvSAeDBQh50r18OOOIoUyCbbAfHq-W"
            st.markdown(f"Info based on documents under [google drive]({g_link})")
            if len(contexts)>0:
                # show reference documents                    
                for item in source_and_articles:
                    st.write(item)
                
        end = datetime.now()
        st_write_speed(start,end)

        st.session_state.messages.append({"role": "assistant", "content": f"{response}"})
    
    # feedback related flow
        response_time_s = (end-start).seconds
        
        #write to session
        if response:
            incr_run_id()
            update_session_state(model_option,
                         department_option,
                         pdfloader_option,
                         chunk_option,
                         refined_question,
                         response,
                         response_time_s
                         )
             
    if st.session_state.run_id>0:
        print("get feedback")
        #set feedbackid
        uid = st.session_state.uuid
        run_id = st.session_state.run_id
        feedback_id = f"{uid}_{run_id}"
        print("feedback_bf",feedback_id)
        
        st.session_state.feedback = streamlit_feedback(
            feedback_type="faces",
            # optional_text_label="[Optional] Please provide an explanation",
            key=feedback_id,
        )
            
        scores = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
        if st.session_state.feedback:
            score = scores.get(st.session_state.feedback["score"])
            print(score)
            print("feedback_af",st.session_state.run_id)
            st.session_state.feedback_id = st.session_state.run_id
            feedback_dict = get_feedback_dict(score)
            load_json_to_bq(feedback_dict, FEEDBACK_SCHEMA,FEEDBACK_OUTPUT_TABLE)
            # st.write(f"Feedback Score: {score}. Thanks for your feedback.")
        
if __name__ == "__main__":
    main()