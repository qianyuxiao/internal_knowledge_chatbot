# Vector Store Management and Chatbot Web Application

This repository contains scripts and instructions for managing a vector store by topic and running a chatbot web application.

## Vector Store Management

### Adding/Modifying Vector Store by Topic
All Vector Store organized by chatbot-id.

1. **Create Vertex Vector Store:**
    Use src/utils/create_vertex_vector_store.py

   If the `chatbot_id` already exists
   - Undeploy and delete the existing index and endpoint under [here](https://console.cloud.google.com/vertex-ai/matching-engine/indexes?project=ecg-ai-416210).
   - Delete its me-bucket under [gcp bucket](https://console.cloud.google.com/storage/browser?project=ecg-ai-416210&pageState=(%22StorageBucketsTable%22:(%22f%22:%22%255B%255D%22,%22s%22:%5B(%22i%22:%22name%22,%22s%22:%220%22)%5D,%22r%22:30))&prefix=&forceOnBucketsSortingFiltering=true)

2. **Download Parameters:**
   - Download the necessary parameters to your local machine.
     ```bash
     gsutil cp gs://${chatbot_id}/me_parameters/${chatbot_id}_me.json ../vector_store_me_parameters/
     ```

3. **Add Documents to Vector Store:**
   - Execute the `add_docs_to_vectorstore.ipynb` script to add documents to the vector store.


## Chatbot Web Application

### Running the Web Application

1. **Run the Chatbot Web Application:**
   - Execute the following command to run the web application.
     ```bash
     streamlit run app.py args --server.fileWatcherType none
     ```

2. **Handling Failures:**
   - In case the application fails to run, execute the following command to check for processes running on port 8501.
     ```bash
     sudo lsof -i :8502
     ```
   - Identify the PID (Process ID) associated with the application and kill it using:
     ```bash
     sudo kill -9 {pid}
     ```

3. **Reference**
   - [How to use Insurance documents](https://docs.google.com/document/d/1tEfC0ebZDBpDwmxZM5mCYC4dAHOCOp9Sj1ht74APxI4/edit)
---

Feel free to customize it further according to your specific needs and details!