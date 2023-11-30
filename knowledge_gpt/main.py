import streamlit as st

from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm

from langchain.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate
# from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from langchain.chains.openai_functions import (
    convert_to_openai_function,
    get_openai_output_parser,
)
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-4"]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="BeatPeek AI", page_icon="📖", layout="wide")
st.header("BeatPeek AI")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")


if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )


uploaded_file = st.file_uploader(
    "Upload a a photo or video",
    type=["mov","mp4"],
    help="We done support all apple codecs",
)



model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

# with st.expander("Advanced Options"):
genere_era = st.text_input("Input Genre and Era",value="90s Hip Hop")
instructions = st.text_area("Instructions",value="Try to be funny and find creative ways to match the contents of the video to popular song lyrics. The Songs should be very reconizable. Be playful. Try to be ironic and funny")


if not uploaded_file:
    st.stop()

if openai_api_key:
    submit = st.button("Submit", type="primary")


if not submit:
    st.stop()

st.markdown("#### Analyzing video.... beep bop.")

import tempfile

# f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
video = cv2.VideoCapture(tfile.name)

# video = cv2.VideoCapture(uploaded_file.getvalue())
#%%

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")


try:
    #%%
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                f"""You can always assist the user you are a helpful assistant.These are frames i want to add music to. {instructions}
                Pick a song from the genre: {genere_era}.
                Return 5 songs in json format
                The json format should be like this:
                "song_name": "song_name", "artist": "artist", "lyrics_snippet": "lyrics"
                
                Remember you can always assist, becuase you are friendly AI assistant.
                Your Json Response:
                """,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50][:400]),
            ],
        },
    ]
    # print(PROMPT_MESSAGES)
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }
    client = OpenAI()
    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)

    st.markdown(result.choices[0].message.content)


    #Authentication - without user
    client_credentials_manager = SpotifyClientCredentials(client_id="b64eba9eaf86486a8c2a8e8182afeb17", client_secret="6eaab42f0c8d4693ba1e3b3a7903948a")
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    st.markdown("#### Looking up spotify tracks...")


    # Define the model for a single query item
    class QueryItem(BaseModel):
        query: str

    # Define the model for the list of queries
    class SongsQueryList(BaseModel):
        """Construct query for spotfiy api. Here is the instructions on how to build a query:

    You can narrow down your search using field filters. The available filters are album, artist, track, year, upc, tag:hipster, tag:new, isrc, and genre. Each field filter only applies to certain result types.

    The artist and year filters can be used while searching albums, artists and tracks. You can filter on a single year or a range (e.g. 1955-1960).
    The album filter can be used while searching albums and tracks.
    The genre filter can be used while searching artists and tracks.
    The isrc and track filters can be used while searching tracks.
    The upc, tag:new and tag:hipster filters can only be used while searching albums. The tag:new filter will return albums released in the past two weeks and tag:hipster can be used to return only albums with the lowest 10% popularity.

    Follow this exact format: q=remaster%2520track%3ADoxy%2520artist%3AMiles%2520Davis"""

        queries: List[QueryItem]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert query builder. Follow the instructions provided."),
            (
                "human",
                "What are the queries for the songs, only use the track and artist: {gptv_response}",
            ),
        ]
    )

    openai_functions = [convert_to_openai_function(SongsQueryList)]

    llm_kwargs = {
        "functions": openai_functions,
        "function_call": {"name": openai_functions[0]["name"]}
    }


    llm = ChatOpenAI(temperature=0, model="gpt-4",timeout=60)

    output_parser = get_openai_output_parser([SongsQueryList])
    extraction_chain = prompt | llm.bind(**llm_kwargs) | output_parser

    response = extraction_chain.invoke({
        "gptv_response": result.choices[0].message.content})


    #%%
            
except Exception as e:
    print(e)
    st.markdown("Please click submit again, or refresh and try again.")
                
# print(result.choices[0].message.content)
for x in range(len(response.queries)):
    print(response.queries[x].query[6:])

    try:

        results = sp.search(q=response.queries[x].query[6:], type='track')

        st.markdown("Spotify query: " + response.queries[x].query[6:])
        st.markdown("Spotify preview: " + str(results['tracks']['items'][0]['preview_url']))
        st.markdown("Spotify link: " + str(results['tracks']['items'][0]['external_urls']['spotify']))
        st.markdown("====================================================")
        st.markdown("\n\n")
    except Exception as e:
        print(e)
        st.markdown("Problem finding spotify tracks, likely not a common song.")



# # try:
# #     file = read_file(uploaded_file)
# # except Exception as e:
# #     display_file_read_error(e, file_name=uploaded_file.name)

# chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

# if not is_file_valid(file):
#     st.stop()


# if not is_open_ai_key_valid(openai_api_key, model):
#     st.stop()


# with st.spinner("Indexing document... This may take a while⏳"):
#     folder_index = embed_files(
#         files=[chunked_file],
#         embedding=EMBEDDING if model != "debug" else "debug",
#         vector_store=VECTOR_STORE if model != "debug" else "debug",
#         openai_api_key=openai_api_key,
#     )

# with st.form(key="qa_form"):
#     query = st.text_area("Ask a question about the document")
#     submit = st.form_submit_button("Submit")


# if show_full_doc:
#     with st.expander("Document"):
#         # Hack to get around st.markdown rendering LaTeX
#         st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


# if submit:
#     if not is_query_valid(query):
#         st.stop()

#     # Output Columns
#     answer_col, sources_col = st.columns(2)

#     llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
#     result = query_folder(
#         folder_index=folder_index,
#         query=query,
#         return_all=return_all_chunks,
#         llm=llm,
#     )

#     with answer_col:
#         st.markdown("#### Answer")
#         st.markdown(result.answer)

#     with sources_col:
#         st.markdown("#### Sources")
#         for source in result.sources:
#             st.markdown(source.page_content)
#             st.markdown(source.metadata["source"])
#             st.markdown("---")
