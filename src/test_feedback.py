import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
 
#
def incr_run_id():
    st.session_state.run_id +=1
           
#
def get_response():
    return "Hello, how are you?"

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'feedback' not in st.session_state:
    print("Init Feedback")
    st.session_state.feedback = dict()

if "run_id" not in st.session_state:
    print("Init run_id")
    st.session_state.run_id = 0

if "uuid" not in st.session_state:
    print("Init uuid")
    st.session_state.uuid = uuid.uuid4()
    
def main():
    question_example = "Saisissez votre question ici, par exemple : Ce qui n'est pas inclus dans le prix de la réservation ?"
    if question_prompt := st.chat_input(question_example):
        st.session_state.messages.append({"role": "user", "content": question_prompt})
        with st.chat_message("user"):
            st.markdown(question_prompt)
        
    
        with st.chat_message("assistant"):
            response = get_response()
            if response:
                st.write(response)
                incr_run_id()
                print(st.session_state.run_id)
                

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
            st.write(score)
            print("feedback_af",st.session_state.run_id)
            st.session_state.feedback_id = st.session_state.run_id
                    
                

if __name__ == "__main__":
    main()
