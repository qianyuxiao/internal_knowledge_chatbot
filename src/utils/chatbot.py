import os 

from utils.doc_to_vertex_search import get_embeddings,get_me_parameters,get_vector_store

#model for question refiner
from vertexai.language_models import TextGenerationModel

from langchain.chat_models import ChatVertexAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


def print_debug(msg):
    print("*"*100)
    print(msg)
    print("*"*100)
    
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
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0,
    max_output_tokens=256,
    top_p=0.7,
    top_k=20
    )
    return response.text

def get_llm(model_id: str,
            max_output_tokens: int=1014,
            temperature: float=0.2,
            top_p: float=0.75,
            top_k: float=40        
            ):
    
    llm = ChatVertexAI(model_name=model_id,
                max_output_tokens=max_output_tokens,
                temperature = temperature,
                top_p=top_p,
                top_k=top_k,
                verbose=False
            )
    return llm

def get_vector_store_by_topic(topic : str            
                 ):   
   
     # Get the directory of the current script
    script_dir = os.getcwd()
    # print_debug(script_dir)
    # If the script directory is not under "src", move it to "src"
    if not script_dir.endswith("src"):
        script_dir = ".."
    # print_debug(script_dir)
    # Set parameters file path based on the script directory
    if topic == "DPO":
        file_path = os.path.join(script_dir, "vector_store_me_parameters", "ecg_dpo_me.json")
    elif topic == "HR":
        file_path = os.path.join(script_dir, "vector_store_me_parameters", "ecg_hr_me.json")
    else:
        file_path = os.path.join(script_dir, "vector_store_me_parameters", "ecg_assurance_me.json")

    
    #Set parameters
    parameters =  get_me_parameters(file_path)

    #Get embeddings
    embeddings = get_embeddings()
    
    #Get vector store
    me = get_vector_store(parameters,embeddings)
    
    return me

def get_template(model_option):
    '''
    Generate prompt template for ConversationChain based on model.
    '''
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    if model_option == "chat-bison":
        system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""
            You are an intelligent assistant helping the users with their questions.
            
            Strictly Use ONLY the provided context to answer the question at the end. Think step-by-step and then answer.
        
            Response style:
            - Always citing the document title that you are generating answer from.
            
            Do not try to make up an answer:
            - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that based on documents that I have got from google drive below."
            - If the context is empty, just  say "I cannot determine the answer to that based on documents that I have got from google drive below."
            - If it's not a question, just say "Do you want to ask a question?"
            """
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
            )
    else:
        prompt_template = ChatPromptTemplate.from_messages(
            [ MessagesPlaceholder(variable_name="history"), human_msg_template]
            )
    return prompt_template

def get_memory():
    memory = ConversationBufferWindowMemory(k=3,
                                            return_messages=True)
    return memory

def get_qa(
    refined_question,
    context,
    buffer_memory,
    prompt_template,
    llm       
        ):
    conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=False)
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{refined_question}")
    
    return response