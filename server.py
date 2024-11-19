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

class CreateQuestionsRAG(BaseModel):
    jobDescription: str
    skills: str
    jobTitle: str
    functionalQuestions: int
    situationalQuestion: int
    behavioralQuestion: int

class Message(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/generate-rag-questions")
async def generate_rag_questions(request: CreateQuestionsRAG):
    jobDescription = request.jobDescription
    skills = request.skills
    jobTitle = request.jobTitle
    functionalQuestions = request.functionalQuestions
    situationalQuestion = request.situationalQuestion
    behavioralQuestion = request.behavioralQuestion
    
    prompt = f""" Write me the questions you would like to ask the candidates for the following job description: {jobDescription}. The skills required are : {skills}. The job title is {jobTitle}. Give me {functionalQuestions} functional questions, {situationalQuestion} situational questions and {behavioralQuestion} behavioral questions to ask the candidates, so the total no. of questions should be {functionalQuestions+situationalQuestion+behavioralQuestion}. Just return me the questions with nothing else and no other text. Just return me the questions. Your response should be in the following format:
    question 1\nquestion2\nquestion3\n.....
    """
    
    try:
        response = query_bedrock_knowledge_base(prompt)
        response_final = remove_empty_strings(response)
        return {"message": response_final}
    except Exception as e:
        print(f"Error querying Bedrock: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

@app.post("/send-message")
async def send_message(request: Message):
    try:
        message = request.message.lower()
        
        if "with answers" in message or "questions and answers" in message:
            num_questions = extract_number_from_query(message) or 5
            subject_area = extract_subject_area(message)
            
            prompt = f"""Generate {num_questions} {subject_area} questions with detailed answers.
            Format your response exactly like this, with clear line breaks between Q/A pairs:
            Q1: What is supervised learning?
            A1: Supervised learning is a type of machine learning where the model learns from labeled data to make predictions.

            Q2: What is clustering?
            A2: Clustering is an unsupervised learning technique that groups similar data points together based on their characteristics.

            Please provide exactly {num_questions} questions and answers following this exact format."""
            
            response = query_bedrock_knowledge_base(prompt)
            qa_pairs = {}
            current_question = None
            
            for line in response:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('Q'):
                    current_question = line.split(':', 1)[1].strip() if ':' in line else line
                elif line.startswith('A') and current_question:
                    answer = line.split(':', 1)[1].strip() if ':' in line else line
                    qa_pairs[current_question] = answer
                    current_question = None
            
            if len(qa_pairs) != num_questions:
                print(f"Warning: Expected {num_questions} Q/A pairs but got {len(qa_pairs)}")
                
            return qa_pairs
            
        else:
            num_questions = extract_number_from_query(message) or 4
            subject_area = extract_subject_area(message)
            
            prompt = f"""Generate {num_questions} {subject_area} questions.
            Return exactly {num_questions} questions, one per line, numbered as:
            1. [question]
            2. [question]
            ...
            {num_questions}. [question]"""
            
            response = query_bedrock_knowledge_base(prompt)
            questions = {}
            
            for i, question in enumerate(response, 1):
                clean_question = question.strip()
                if '. ' in clean_question:
                    clean_question = clean_question.split('. ', 1)[1]
                if clean_question:
                    questions[f"q{i}"] = clean_question
                
                if len(questions) >= num_questions:
                    break
                    
            if len(questions) != num_questions:
                print(f"Warning: Expected {num_questions} questions but got {len(questions)}")
                
            return questions

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

def extract_number_from_query(query: str) -> int:
    words = query.split()
    for _, word in enumerate(words):
        if word.isdigit():
            return int(word)
    return None

def extract_subject_area(query: str) -> str:
    """Extract the subject area from the query."""
    query = query.lower()
    query = query.replace("give me", "")
    query = query.replace("with answers", "")
    query = query.replace("questions", "")
    query = query.replace("and answers", "")
    
    words = [word for word in query.split() if not word.isdigit()]
    
    subject_area = " ".join(words).strip()
    return subject_area if subject_area else "general knowledge"

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
                output_list = output_text.split('\n')
                return [line for line in output_list if line.strip()]
            else:
                raise ValueError('No text in the response')
        else:
            raise ValueError('No results found in the response')
    except ClientError as e:
        print(f"Error invoking Bedrock knowledge base: {e}")
        raise

def remove_empty_strings(input_list):
    return list(set(item for item in input_list if item))

if __name__ == '__main__':
    uvicorn.run("server:app", port=8000, reload=True)