# from (file) import function
from fastapi import FastAPI #import
from chatbot import get_response

app = FastAPI ()

@app.get("/")
def root():
    return {"message": "T2B ON"}

@app.post("/chat") #post = sends data, /chat = url path
def chat(question: str): #func name (parameter:type)
    response = get_response(question)
    return {"response": response}

