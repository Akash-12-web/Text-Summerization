import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

## streamlit app
st.set_page_config(page_title="Text Summarization App")
st.title("Text Summarization App");
st.subheader("Enter a URL to summarize the content")


## get groq api key
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key",type="password",value="")

## get url input
url = st.text_input("URL",label_visibility="collapsed")

## Gemma model is used for summarization, you can change the model to any other model available in Groq
llm = ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant",temperature=0.7)
prompt_template = """Write a concise summary of the following: 
The summary must be written entirely in English:
content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
if st.button("Summarize"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter both the Groq API Key and the URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading content..."):
                ## load the content from the url
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url)
                    
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
                data = loader.load()

                ## chain summarization
                chain = load_summarize_chain(llm,chain_type="refine",verbose=True,question_prompt=prompt)
                output_sum = chain.run(data)
                st.subheader("Summary:")
                st.success(output_sum)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

