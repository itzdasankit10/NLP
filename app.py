import asyncio
import re
from io import BytesIO
from typing import List, Dict, Any

import ollama
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# --- App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading (Consolidated and CPU-forced) ---
print("Configuring all Hugging Face models to use device: cpu")
aclient = ollama.AsyncClient()
qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device='cpu')
st_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')


# --- Pydantic Models for API Requests & Responses ---
class ContextRequest(BaseModel):
    context: str

class QuestionItem(BaseModel):
    question: str
    answer: str | None = None

class AnswerRequest(BaseModel):
    context: str
    questions: Dict[str, List[QuestionItem]]

class EvaluationRequest(BaseModel):
    context: str
    question: str
    generated_answer: str
    q_type: str 

class PdfRequest(BaseModel):
    answered_questions: Dict[str, List[QuestionItem]]


# --- API Endpoints ---
@app.post("/generate-questions")
async def generate_questions(req: ContextRequest):
    ctx = req.context
    if not ctx:
        raise HTTPException(status_code=400, detail="No context provided.")

    p = f"""
You are an assistant that generates a three-part quiz from the provided text. Follow these instructions exactly.

**Input Text:**
---
{ctx}
---

**Instructions:**
Generate a quiz with exactly three distinct parts below. Use the specified headers for each part. Do not add any extra conversation, introductions, or concluding text.

**Part 1: Multiple Choice Questions**
Generate exactly 5 MCQs. Each question must have four options (A, B, C, D).
Do NOT include the correct answer.

**Part 2: Fill in the Blanks**
Generate exactly 5 fill-in-the-blanks questions. Replace a key word or phrase with '____'.
The question must be a complete sentence. Do NOT include the answer.

**Part 3: Subjective Questions**
Generate exactly 5 subjective (long-answer) questions that encourage critical thinking about the text.

**IMPORTANT:** You must generate all three parts and use the exact "Part X:" headers. Do not stop generating until all three sections are fully complete.
"""
    try:
        r = await aclient.generate(model="llama3.1", prompt=p)
        t = r["response"].strip()
        
        d = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}
        parts = re.split(r'part\s+\d+:', t, flags=re.IGNORECASE)

        if len(parts) > 1:
            c = parts[1]
            c = re.sub(r'.*multiple\s+choice\s+questions.*?\n', '', c, flags=re.IGNORECASE).strip()
            q_blocks = re.split(r'\n\s*(?=\d+\.\s*|Q:)', c)
            d["mcqs"] = [{"question": q.strip(), "answer": ""} for q in q_blocks if q.strip()]

        # --- THIS IS THE CORRECTED PART ---
        if len(parts) > 2:
            c = parts[2]
            c = re.sub(r'.*fill\s+in\s+the\s+blanks.*?\n', '', c, flags=re.IGNORECASE).strip()
            
            # New robust logic: Split only at the start of a new numbered question
            q_blocks = re.split(r'\n\s*(?=\d+\.)', c)
            
            temp_list = []
            for block in q_blocks:
                # Join multi-line questions and clean up whitespace
                clean_q = " ".join(block.strip().splitlines()).strip("0123456789. ").strip()
                if clean_q:
                    temp_list.append({"question": clean_q, "answer": ""})
            d["fill_in_the_blanks"] = temp_list
        # --- END OF CORRECTION ---

        if len(parts) > 3:
            c = parts[3]
            c = re.sub(r'.*subjective\s+questions.*?\n', '', c, flags=re.IGNORECASE).strip()
            # This logic is already robust, so no changes are needed here.
            q_blocks = re.split(r'\n\s*(?=\d+\.)', c)
            temp_list = []
            for block in q_blocks:
                clean_q = " ".join(block.strip().splitlines()).strip("0123456789. ").strip()
                if clean_q:
                    temp_list.append({"question": clean_q, "answer": ""})
            d["subjective"] = temp_list
        
        if not all(d.values()):
             raise HTTPException(status_code=500, detail="AI model failed to generate all quiz parts. Please try again.")

        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

@app.post("/generate-answers")
async def generate_answers(req: AnswerRequest):
    ctx = req.context
    q_data = req.questions
    answered_data = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    for q_type, q_list in q_data.items():
        if not q_list:
            continue

        prompts = []
        for i, item in enumerate(q_list):
            prompts.append(f"{i+1}. {item.question}")
        joined_prompts = "\n".join(prompts)
        
        p_template = ""
        if q_type == "mcqs":
            p_template = f"Context:\n{ctx}\n\nFor each question below, provide ONLY the correct option letter and text (e.g., 'C) Paris'), each on a new line.\n\n{joined_prompts}"
        elif q_type == "fill_in_the_blanks":
            p_template = f"Context:\n{ctx}\n\nFor each question below, provide ONLY the single word or short phrase that fits in the blank, each on a new line.\n\n{joined_prompts}"
        else:
             p_template = f"Context:\n{ctx}\n\nAnswer each of the following questions in detail. Start each answer with the corresponding number (e.g., '1. [Answer text]').\n\n{joined_prompts}"

        r = await aclient.generate(model="llama3.1", prompt=p_template)
        raw_answers = r["response"].strip()
        
        raw_answers = re.sub(r'^(.*?)(?=1\.)', '', raw_answers, flags=re.DOTALL)
        split_answers = re.split(r'\n\s*(?=\d+\.)', raw_answers)
        
        temp_list = []
        for i, item in enumerate(q_list):
            new_item = item.model_copy()
            if i < len(split_answers) and split_answers[i].strip():
                answer_text = re.sub(r'^\d+\.\s*', '', split_answers[i].strip())
                new_item.answer = answer_text
            else:
                new_item.answer = "No answer generated."
            temp_list.append(new_item)
        
        answered_data[q_type] = temp_list
        
    return answered_data


@app.post("/evaluate-accuracy")
async def evaluate_accuracy(req: EvaluationRequest):
    if not req.context:
        raise HTTPException(status_code=400, detail="No context is set.")

    if req.q_type == "subjective":
        emb1 = st_model.encode(req.context, convert_to_tensor=True)
        emb2 = st_model.encode(req.generated_answer, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2)
        score = sim.item()
        return {
            "generated_answer": req.generated_answer,
            "bert_answer": "N/A (Relevance Score)",
            "similarity_score": score
        }
    else:
        loop = asyncio.get_running_loop()
        bert_result = await loop.run_in_executor(
            None, qa_pipe, req.question, req.context
        )
        bert_answer = bert_result["answer"]

        emb1 = st_model.encode(req.generated_answer, convert_to_tensor=True)
        emb2 = st_model.encode(bert_answer, convert_to_tensor=True)
        
        sim = util.pytorch_cos_sim(emb1, emb2)
        score = sim.item()
        return {
            "generated_answer": req.generated_answer,
            "bert_answer": bert_answer,
            "similarity_score": score
        }
    

@app.post("/answer-questions-pdf")
async def answer_questions_pdf(req: PdfRequest):
    data = req.answered_questions
    if not any(data.values()):
        raise HTTPException(status_code=400, detail="No answered questions provided.")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Generated Answers", styles['h1']))
    story.append(Spacer(1, 24))

    for q_type, q_list in data.items():
        if q_list:
            header = q_type.replace('_', ' ').title()
            story.append(Paragraph(header, styles['h2']))
            for i, item in enumerate(q_list, 1):
                q_text = f"{i}. <b>{item.question}</b>".replace('\n', '<br/>')
                story.append(Paragraph(q_text, styles['Normal']))
                story.append(Spacer(1, 6))
                a_text = (item.answer or "").replace('\n', '<br/>')
                story.append(Paragraph(a_text, styles['Normal']))
                story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=answers_script.pdf"
    })