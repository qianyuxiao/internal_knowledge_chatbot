import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
import datetime as dt
from google.cloud import bigquery
from utils.bq_utils import load_json_to_bq
#
def incr_run_id():
    st.session_state.run_id +=1
           
#
def get_response():
    return "Hello, how are you?"

output_table_name = "ecg-big-data-sandbox.qianyu_test.chatbot_evaluation_test"
model_option = "llamma"
app_type ="feedback_test"


# get feedbacks related settings
def init_session_state():
    st.session_state.messages = []
    st.session_state.feedback = dict()
    st.session_state.uuid = str(uuid.uuid4())
    st.session_state.run_id = 0
    st.session_state.question_prompt = None
    st.session_state.response = None
    st.session_state.response_time_s = None
    st.session_state.model_id = None


def get_feedback_dict(
                      score):
    feedback_dict = {
        "app_type": app_type,
        "uuid": st.session_state.uuid,
        "run_id": st.session_state.run_id,
        "feedback_time":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": st.session_state.model_id,
        "question": st.session_state.question_prompt,
        "response": st.session_state.response,
        "response_time_s": st.session_state.response_time_s,
        "score": score
    }
    return feedback_dict

schema = [
    bigquery.SchemaField("uuid", "STRING"),
    bigquery.SchemaField("run_id", "INT64"),
    bigquery.SchemaField("app_type", "STRING"),
    bigquery.SchemaField("model_id", "STRING"),
    bigquery.SchemaField("feedback_time", "TIMESTAMP"),
    bigquery.SchemaField("response_time_s", "INT64"),
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("response", "STRING"),
    bigquery.SchemaField("score", "FLOAT64")
]


if "uuid" not in st.session_state:
    init_session_state() 

def main():
    
    question_example = "Saisissez votre question ici, par exemple : Ce qui n'est pas inclus dans le prix de la réservation ?"
    if question_prompt := st.chat_input(question_example):
        st.session_state.messages.append({"role": "user", "content": question_prompt})
        with st.chat_message("user"):
            st.markdown(question_prompt)    
    
        with st.chat_message("assistant"):
            start = dt.datetime.now()
            response = get_response()
            end  = dt.datetime.now()
            response_time_s = (end-start).seconds 
            
            if response:
                st.write(response)
                incr_run_id()
                print(st.session_state.run_id)
                
                #update session state
                st.session_state.model_id = model_option
                st.session_state.question_prompt = question_prompt
                st.session_state.response = response
                st.session_state.response_time_s = response_time_s
                

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
            load_json_to_bq(feedback_dict, schema,output_table_name)
            # st.write(f"Feedback Score: {score}. Thanks for your feedback.")
                    
                

if __name__ == "__main__":
    main()
