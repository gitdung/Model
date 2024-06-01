import os 
from dotenv import load_dotenv
import sys

load_dotenv()
sys.path.insert(1, os.getenv('PATH_ROOT'))


from fastapi import FastAPI 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from main.embeddingURL.predict import *
from main.processing_data.getLexicalFeature import *
from main.featureURL.lexical_feature import *




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelName(str, Enum):
    phobert = 'Model PhoBert'
    xml_roberta = 'Model XML Roberta'

class DomainRequest(BaseModel):
    domain: str
    model_name : ModelName

class DomainResponse(BaseModel):
    domain: str
    entropy: float
    percentageDigits: float
    domainLength: int
    specialChars: int
    result: str

@app.post("/api/infer", response_model=DomainResponse)
async def infer_domain(request: DomainRequest):
    print(f'''
----------------------------------------------------------------------------------------------------------------
API CALL
domain: {request.domain}
model_name : {request.model_name}
''')
    domain = request.domain
    model_name = request.model_name
    lexical = LexicalURLFeature(domain)
    
    # validate domain
    if len(domain) < 3 or domain is None:
        raise JSONResponse(
            status_code=400,
            content={"error": "Domain không hợp lệ", "status": 400}
        )
    
    if model_name not in ['Model PhoBert', 'Model XML Roberta']:
        raise JSONResponse(
            status_code=400,
            content={"error": "Domain không hợp lệ", "status": 400}
        )     


    response = DomainResponse(
        domain=domain,
        entropy=lexical.get_entropy(),
        percentageDigits=lexical.get_percentage_digits(),
        domainLength=lexical.get_length_url(),
        specialChars=lexical.get_count_special_characters(),
        result=detect_toxic_website(domain, model_name)
    )
    
    
    print(f'''
API RESPONSE
domain: {response.domain}
entropy: {response.entropy}
percentageDigits: {response.percentageDigits}
domainLength: {response.domainLength}
specialChars: {response.specialChars}
result: {response.result}
---------------------------------------------------------------------------------------------------------------
          ''')
    
    #ghi log
    with open('log.txt', 'a' , encoding='utf-8') as f:
        f.write(f'''
----------------------------------------------------------------------------------------------------------------
API CALL
domain: {request.domain}
model_name : {request.model_name}      
      
API RESPONSE
domain: {response.domain}
entropy: {response.entropy}
percentageDigits: {response.percentageDigits}
domainLength: {response.domainLength}
specialChars: {response.specialChars}
result: {response.result}
---------------------------------------------------------------------------------------------------------------
''')
    
    return JSONResponse(
            status_code=200,
            content={
                "data" : response.dict(),
                "status": 200
            }
        )

if __name__ == "__main__":
    import uvicorn
    print(f"Server is running at http://{os.getenv('HOST')}:{os.getenv('SERVER_PORT')}/docs")
    uvicorn.run(app, host=os.getenv('HOST'), port=int(os.getenv('SERVER_PORT')))