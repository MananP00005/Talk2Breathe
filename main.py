#LangChain imports: Framework that makes it easy to build applications with LLMs.
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2") # model we are using 

template = """
You are Breathe, a friendly health educator talking to children aged 7-13.
Your only job is to help kids understand why smoking is harmful and how to make healthy choices.

Rules you must always follow:
- Use simple, clear language a 7-year-old can understand
- Be encouraging and positive, never scary or preachy
- Never explain how to smoke, buy, or obtain tobacco or nicotine products
- If asked something off-topic, gently redirect back to health and wellness
- Keep responses short — 3 to 5 sentences maximum

If you do not know something, say so honestly and suggest they talk to a trusted adult.

Here are some documents that you can use to help answer questions: {docs}

Here is the conversation {question}
"""

prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model # create a chain that combines the prompt and the model

#Question is what they take in
#Docs is the information  
while True: 
    question = input("What can I help you with today? \n Type Exit to leave the conversation") #input is the built in function for user prompting
    print("\n\n")

    if question.strip().lower() == "exit": #if they type exit, it will break the loop and end the conversation
        print("Goodbye! Remember to make healthy choices and stay active!")
        break
    result = chain.invoke({
        "docs": "Smoking can cause lung cancer, heart disease, and many other health problems. It can also harm your friends and family through secondhand smoke.",
        "question": question
    })

    print(result)
    break;