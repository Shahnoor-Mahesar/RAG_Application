from pdf_extract import get_pdf_content,get_chunks,decode_chunk
from model import Model
from process import process_and_store_text, query_text


# model=Model(key='',model='llama3-8b-8192')


# answer= model.prompt(question="what is the age of elon musk and zuck")

# print(answer)


long_text="hello this is shahnoor mahesar from sukkur"

print(get_chunks(long_text))
print(decode_chunk(get_chunks(long_text)[0]))

text=get_pdf_content('/home/user/rag-app/resources/javascript_tutorial.pdf')
# print(text)


# Process the text and store embeddings in ChromaDB
process_and_store_text(text)

# Example query
query_text_str = "how to write conditions(if,else) in javascript."


nearest_neighbors = query_text(query_text_str)
print("Nearest neighbors:", nearest_neighbors)
