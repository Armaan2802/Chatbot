# import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.prompt import PromptTemplate
import os
# -------------------------------------------------------------------------

print("All imports are importing")

# openai_api_key = os.environ.get("OPENAI_API_KEY")

def get_document():
    loader = CSVLoader(file_path='data.csv')
    data1 = loader.load()
    loader2= CSVLoader(file_path='BlogLinks.csv')
    data2 = loader2.load()
    return data1

my_data = get_document()

words=["hi", "hii", "hiii", "hey", "heyy", "heyyy", "hello", "helloo", "hellooo", "g morning", "gm", 
    "gmorning", "good morning", "goodmorning", "morning","mornin", "good day", "good afternoon", 
    "good evening", "greetings", "greeting", 
    "good to see you", "its good seeing you", "how are you", "how're you", 
    "how are you doing", "how ya doin'", "how ya doin", "how is everything", 
    "how is everything going", "how's everything going", "how is you", "how's you", 
    "how are things", "how're things", "how is it going", "how's it going", "how's it goin'", 
    "how's it goin", "how is life been treating you", "how's life been treating you", 
    "how have you been", "how've you been", "what is up", "what's up", "what is cracking", 
    "what's cracking", "what is good", "what's good", "what is happening", "what's happening", 
    "what is new", "what's new", "what is neww", "g'day", "howdy",
    "who are you", "who you", "namaste","who r u", "menu"]

text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
my_doc = text_splitter.split_documents(my_data)

embeddings = OpenAIEmbeddings(openai_api_key = [OPEN_API_KEY])
vectordb = Chroma.from_documents(my_doc, embeddings)

template = open("instruct.txt").read()

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def generate_response(query,chat_history):
        if query:
            llm = OpenAI(temperature=0.7 , model_name="gpt-3.5-turbo")
            my_qa = ChatVectorDBChain.from_llm(llm, vectordb, qa_prompt=QA_PROMPT, return_source_documents=True)
            result = my_qa({"question": query, "chat_history": chat_history})

        return result["answer"]

print("All good before chatbot begins")

#Beginning of conversation processing

def hello_http(s):
    print("s function working")

    def generate_history(l):
        input1= list()
        history = list()
        for i in l:
            his = i.get("output")
            inp1 = i.get('input')
            input1.append(inp1)
            history.append(his)
        hist = list(zip(input1, history))
        if len(l)==0 :
            hist = list()
        return hist
    print("Receiving s.json")
    print("Incoming JSON: ", s.json)
    print("Received s.json")
    data  = s.json
    data_ = data.get('body')
    message = data_.get('input')
    message = str(message)
    print(f'Message: {message}')
    history_ = data_.get('history')
    history_ = generate_history(history_)
    print(f'History: {history_}')
    print("Hisotry Length: ", len(history_))
    user=data_.get('user')
    user=str(user)
    print("Number: ", user)

    
    def my_chatbot(input, history):
        history = history
        lowin=input.lower().strip( )
        crt=lowin
        output=''

        for el in words:
            if lowin==el:
                output='welcome2'
                break

        if crt=='information':
            output='information'
        
        if crt=='ayurvedic ingredients':
            output='ingredients'
        
        if crt=="kapiva's offerings":
            output='offerings'

        if crt=="kapiva's health range":
            output='healthrange'

        if crt=="health programmes":
            output='programmes'

        if crt=="health concerns":
            output='health_concerns'
        
        if crt=='enquiry':
            output='enquiry'
        
        if crt=='track order':
            output='trackorder'

        if crt=='customer support':
            output='cs'

        if crt=='thank you' or crt=='thanks' or crt=='ty' or crt=='thankyou':
            output = 'thankyou'

        if crt=='bye'or crt=='goodbye' or crt=='bye bye':
            output='goodbye'
        
        if lowin =='clear':
            output='clear'

        print("Output: ", output)

        
        if len(output)!=0:
            payload={ 
            "messaging_product": "whatsapp",
            "to": user,
            "type": "template",
            "template": {
                "name": output,
                "language":{
                    "code":"EN"
                }
            }
            }

            print("Generated template body")

        else:
            output2 = generate_response(input,history)
            output2=str(output2)+"\n\nPlease type 'clear' to start a new conversation"
            # output2="We are currently under maintenance ⚙️, please check back with us shortly!\n\nApologies for any inconvenience caused."
            print("Output2: ", output2)

            payload={
                "messaging_product": "whatsapp",
                "to": user,
                "type": "text",
                "text": {
                    "body": output2
                }
            }

            print("Generated text body")

        print("\n")    
        return payload

    x = my_chatbot(message, history_)
    print(x)
    print("Executed Successfully")
    return x
    
