import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

with st.sidebar:
    st.title("langchain")
    st.markdown("""
                about
                streamlit
                langchain
                opanai
            
                """)
    add_vertical_space(5)
    st.write("made by tahir")


def main():

    st.header("Chat with Your PDF")
    load_dotenv()
    pdf = st.file_uploader("upload you pdf", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]

        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstote = pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            embaddings = OpenAIEmbeddings()
            vectorstote = FAISS.from_texts(chunks, embedding=embaddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstote, f)
            st.write("Embedding Computation Completed ")

        query = st.text_input("Ask Questions from this file:")
        # st.write(query)

        if query:
            docs = vectorstote.similarity_search(query=query)
            llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write(cb)
            st.write(response)
            # st.write(docs)


if __name__ == "__main__":
    main()