from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# load environment variables from a .env file, api key
load_dotenv()

# create llm object
llm = ChatOpenAI(model = "gpt-4", temperature = 0)

# create prompt templates
prompt = ChatPromptTemplate.from_messages([
    ("user", "{question}"), 
])
question_input = input("enter your code:")
final_prompt = prompt.format_messages(question = question_input)

# create chroma client
vector_db = chromadb.Client()

# create embeddings object
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

# create vector store
vectorstore = Chroma(
    collection_name = 'docs', 
    embedding_function = embeddings, 
    client=vector_db
)

# add text to the vector store
vectorstore.add_texts([question_input], ids=["0"])

# create retriever
retriever = vectorstore.as_retriever(
    search_type = 'mmr', 
    search_kwargs = {'k':1, 'lambda_mult':0.8}
)

# chains

# specialized prompt for chain
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI coding assistant. Analyze the user's code snippet for syntax errors, logic issues, or improvements. Then suggest the corrected version. Also make sure to explain what the errors are and why they needed to be changed"), 
    ("user", "Code:\n{context}\nUser Question:{{question}}")
])
chain = create_stuff_documents_chain(llm, review_prompt)

# connect the retriever + chain
retrieval_chain = create_retrieval_chain(retriever, chain)

response = retrieval_chain.invoke({"input": question_input})
print(response["answer"])