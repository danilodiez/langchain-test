from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# temperature in OpenAI is the randomness -> 0 = stable, 1 = random

OPENAI_API_KEY = config('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = config('ACTIVELOOP_TOKEN')


def main():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)

    text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
    print(llm(text))


'''
We want to create a chain that generates a possible
name for a company that produces eco-friendly bottles
'''


def create_name():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    print(chain.run("eco-friendly water bottles"))


def mantain_memory():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )

    # Start the conversation
    conversation.predict(input="Tell me about yourself.")

    # Continue the conversation
    conversation.predict(input="What can you do?")
    conversation.predict(input="How can you help me with data analysis?")

    # Display the conversation
    print(conversation)


def using_deeplake():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    # create our documents
    texts = [
        "Napoleon Bonaparte was born in 15 August 1776",
        "Louis XIV was born in 5 September 1638",
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(texts)

    # create Deep Lake dataset
    my_activeloop_org_id = "danilodiez"
    my_activeloop_dataset_name = "langchain_course"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, token=ACTIVELOOP_TOKEN)

    # add documents to our Deep Lake dataset
    db.add_documents(docs)








# main()
# createName()
# mantainMemory()
using_deeplake()
