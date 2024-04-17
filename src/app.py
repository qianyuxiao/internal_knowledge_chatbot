from datetime import datetime
import time
import streamlit as st
from utils.chatbot import *


def st_write_speed(start,end):
    speed = (end-start).seconds
    st.write(f'{speed}s used')
    return speed

def print_info(msg):
    print("-"*100)
    print(msg)
    

def greet_msg(topic_option):
    greet_msg  = f"Greetings! I'm here to assist you with any inquiries concerning ** {topic_option} **. Should you wish to explore other subjects, feel free to update your topic option on the left-hand side. "
    for word in greet_msg.split():
        yield word + " "
        time.sleep(0.03)

def init_st_session_state(topic_option,
                          model_option,
                          show_ref_content_option):
    print("Initiating session state....")
    st.session_state.messages = []
    st.session_state.topic_option = topic_option
    st.session_state.model_option = model_option
    st.session_state.show_ref_content_option = show_ref_content_option
    st.session_state.memory =  get_memory() 
    st.session_state.memory.clear()
    show_message()
        
def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()
  
# chatbot related cache function    
@st.cache_resource
def load_model(model_option):
    print_info(f"Load model {model_option}")
    llm = get_llm(model_option)   
    return llm

@st.cache_resource
def load_vector_store(topic_option):
    print_info(f"Load vector store for {topic_option}")
    vector_store = get_vector_store_by_topic(topic_option)  
    return vector_store

@st.cache_resource
def load_qa_template(model_option):
    return get_template(model_option)

def reset_vector_store():
    load_vector_store.clear()

def get_google_doc_link(topic_option):
    if topic_option=="DPO":
        g_link = "https://drive.google.com/drive/folders/1aIxinK3LugUGlNIr23mQSTxqn8fSdxsi"
    elif topic_option == "HR":
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
    title = "🤖ECG Internal Knowledge Chatbot"
    st.set_page_config(page_title="ECG Internal Knowledge Chatbot", page_icon="🤖")
    st.title(title)
    
    ## Sidebar
    st.sidebar.title('Usage and Limitation')
    st.sidebar.write("- LLMs generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.")
    st.sidebar.markdown("- You can find the original documents that chatbot is retrieving knowledge from [here](https://drive.google.com/drive/folders/1TtxvSAeDBQh50r18OOOIoUyCbbAfHq-W)")
    
    ### Choice box
    topic_option = st.sidebar.selectbox(
        'Topic',
        ("DPO", "HR","Insurance"))
    
    model_option = st.sidebar.selectbox(
        'Chatbot model',
        ("chat-bison", "gemini-pro" ))
    
    show_ref_content_option = st.sidebar.selectbox(
        'Show Ref. Content?',
        (False, True))
    
    ### SLIDER
    search_distance = st.sidebar.slider('Retriver search distance',0.0, 1.0, 0.75)
    search_nb_docs = st.sidebar.slider('Retriver search maximum docs',1, 8, 4)
    
    ### Buttons
    st.sidebar.button('Reset Chat', on_click=reset_conversation, type="primary")
    st.sidebar.button('Clear Vector Store Cache', on_click=reset_vector_store, type="primary")
    
    # Load and cache chatbot element
    llm = load_model(model_option)
    me = load_vector_store(topic_option)
    prompt_template = load_qa_template(model_option)
    g_link = get_google_doc_link(topic_option)

    
    # Show messages
    if 'messages' not in st.session_state:
        init_st_session_state(topic_option,
                              model_option,
                              show_ref_content_option)
        # memory.clear()
        with st.chat_message("assistant"):
            st.write(greet_msg(st.session_state.topic_option))
    

    show_message()
    
    # Chat flow
    question_example = "Saisissez votre question ici, par exemple : Ce qui n'est pas inclus dans le prix de la réservation ?"
    if question_prompt := st.chat_input(question_example):
        st.session_state.messages.append({"role": "user", "content": question_prompt})
        with st.chat_message("user"):
            st.markdown(question_prompt)
            
        # check if model option changes
        if st.session_state.model_option!=model_option:
            init_st_session_state(topic_option,
                              model_option,
                              show_ref_content_option)
            st.write(f"Clearing cache and changing model to {model_option}...")
            llm = get_llm(model_option)
            prompt_template = load_qa_template(model_option)
            st.session_state.model_option = model_option

        # check if topic change
        if st.session_state.topic_option!=topic_option:
            init_st_session_state(topic_option,
                              model_option,
                              show_ref_content_option)
            st.write(f"Clearing cache and changing model to {topic_option}...")
            me = load_vector_store(topic_option)
            g_link = get_google_doc_link(topic_option)
            st.session_state.topic_option = topic_option

            
        # get answer and stream answer
        start = datetime.now()
        conversation = ConversationChain(memory=st.session_state.memory, prompt=prompt_template, llm=llm, verbose=False)
        
        #get refined question based on conversation history
        if len(st.session_state.messages)>2:
            refined_question = query_refiner(st.session_state.messages,question_prompt)
            st.write(f"Refined question based on conversation history: {refined_question}")
        else:
            refined_question = question_prompt
            
        contexts = me.similarity_search(refined_question, k=search_nb_docs,search_distance=search_distance)
        response = conversation.predict(input=f"Context:\n {contexts} \n\n Query:\n{question_prompt}")
        
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
            
            if len(contexts)>0 and response.strip()!="Do you want to ask a question?":
                # show reference documents
                st.write("-"*100)     
                st.markdown(f"Reference documents under [google drive]({g_link}):")
                for item in source_and_articles:
                    st.write(item)
                
        end = datetime.now()
        st_write_speed(start,end)

        st.session_state.messages.append({"role": "assistant", "content": f"{response}"})

if __name__ == "__main__":
    main()
