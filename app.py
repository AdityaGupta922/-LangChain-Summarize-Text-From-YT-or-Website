import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                    docs=loader.load()
                else:
                    # Use requests and BeautifulSoup to fetch and parse the webpage
                    response = requests.get(generic_url, headers={"User-Agent": "Mozilla/5.0"})
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract text content from the page
                        page_content = ' '.join([p.get_text() for p in soup.find_all('p')])
                        docs = [Document(page_content=page_content, metadata={"source": generic_url})]
                    else:
                        st.error("Failed to fetch the webpage. Please check the URL and try again.")
                        docs = []

                # Check if content is loaded
                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to load content from the URL. Please check the URL and try again.")
                else:
                    ## Chain For Summarization
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    output_summary=chain.run(docs)

                    st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")