import os
import json
import boto3
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from botocore.exceptions import ClientError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

bedrock = boto3.client('bedrock-runtime', 
    region_name=os.getenv('region'),
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key')
)

KNOWLEDGE_BASE_ID = "EY0ZGLB9OT"
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-premier-v1:0"

class Message(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/chat.html", "r") as f:
        return f.read()

@app.post("/send_message")
async def send_message(message: Message):
    try:
        response = query_bedrock_knowledge_base(message.message)
        return {"message": response}
    except Exception as e:
        print(f"Error querying Bedrock: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

def query_bedrock_knowledge_base(query: str):
    try:
        response = bedrock.invoke_model(
            body=json.dumps({
                'inputText': query
            }),
            modelId=MODEL_ARN,
            contentType='application/json',
            accept='application/json',
            trace='DISABLED'
        )
        
        response_body = response['body'].read()
        response_data = json.loads(response_body)
        
        if 'results' in response_data and len(response_data['results']) > 0:
            output_text = response_data['results'][0].get('outputText')
            if output_text:
                return output_text
            else:
                raise ValueError('No text in the response')
        else:
            raise ValueError('No results found in the response')
        
    except ClientError as e:
        print(f"Error invoking Bedrock knowledge base: {e}")
        raise


if __name__ == '__main__':
    uvicorn.run("server:app", port=8000, reload=True)