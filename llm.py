from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

response = llm.stream("Write a poem about AI")
# print(response)

for chunk in response:
    print(chunk.content, end="", flush=True)

