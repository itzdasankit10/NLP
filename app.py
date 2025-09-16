import asyncio
import json
import re
import os
from io import BytesIO
from typing import Dict, List

import ollama
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuring all Hugging Face models to use device: cpu")
a = ollama.AsyncClient()
b = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device='cpu')
c = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

BOOKS_BASE_DIR = "BOOKS"
SUBJECT_ABBREVIATIONS = {
    "Geography": "geo",
    "History": "his",
    "Economics": "eco",
    "English": "eng",
    "Biology": "bio",
    "Political Science": "pol"
}

class QuestionCounts(BaseModel):
    mcqs: int
    fill_in_the_blanks: int
    subjective: int

class QuestionItem(BaseModel):
    question: str
    answer: str | None = None

class QuestionsPayload(BaseModel):
    questions: Dict[str, List[QuestionItem]]

class PdfRequest(BaseModel):
    answered_questions: Dict[str, List[QuestionItem]]

def get_context_from_selection(class_num: str, subject: str) -> str:
    subject_abbr = SUBJECT_ABBREVIATIONS.get(subject)
    if not subject_abbr:
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' is not configured.")

    class_dir = os.path.join(BOOKS_BASE_DIR, f"CLASS {class_num}")
    if not os.path.isdir(class_dir):
        raise HTTPException(status_code=404, detail=f"Directory for Class {class_num} not found.")

    file_prefix = f"class_{class_num}_{subject_abbr}"
    found_file_path = None
    try:
        for filename in os.listdir(class_dir):
            if filename.startswith(file_prefix) and filename.endswith(".txt"):
                found_file_path = os.path.join(class_dir, filename)
                break
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail=f"Base directory '{BOOKS_BASE_DIR}' not found.")

    if not found_file_path:
        raise HTTPException(status_code=404, detail=f"Textbook file for Class {class_num}, Subject {subject} not found.")

    try:
        with open(found_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file '{found_file_path}': {e}")

@app.post("/generate-questions")
async def generate_questions(
    class_num: str = Form(...),
    subject: str = Form(...),
    counts_json: str = Form(...)
):
    ctx = get_context_from_selection(class_num, subject)
    if not ctx:
        raise HTTPException(status_code=400, detail="The selected textbook file is empty.")

    try:
        counts = QuestionCounts.model_validate_json(counts_json)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for question counts: {e}")

    p = f"""
You are an assistant that generates a three-part quiz from the provided text. Follow these instructions exactly.
**Input Text:**
---
{ctx}
---
**Instructions:**
Generate a quiz with exactly three distinct parts below. Use the specified headers for each part. Do not add any extra conversation, introductions, or concluding text.
**Part 1: Multiple Choice Questions**
Generate exactly {counts.mcqs} MCQs. Each question must have four options (A, B, C, D).
Do NOT include the correct answer.
**Part 2: Fill in the Blanks**
Generate exactly {counts.fill_in_the_blanks} fill-in-the-blanks questions. Replace a key word or phrase with '____'.
The question must be a complete sentence. Do NOT include the answer.
**Part 3: Subjective Questions**
Generate exactly {counts.subjective} subjective (long-answer) questions that encourage critical thinking about the text.
**IMPORTANT:** You must generate all three parts and use the exact "Part X:" headers. Do not stop generating until all three sections are fully complete.
"""
    try:
        r = await a.generate(model="llama3.1", prompt=p)
        t = r["response"].strip()
        
        d = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}
        parts = re.split(r'part\s+\d+:', t, flags=re.IGNORECASE)

        if len(parts) > 1:
            content = parts[1]
            content = re.sub(r'.*multiple\s+choice\s+questions.*?\n', '', content, flags=re.IGNORECASE).strip()
            q_blocks = re.split(r'\n\s*(?=\d+\.\s*|Q:)', content)
            d["mcqs"] = [{"question": q.strip(), "answer": ""} for q in q_blocks if q.strip()]

        if len(parts) > 2:
            content = parts[2]
            content = re.sub(r'.*fill\s+in\s+the\s+blanks.*?\n', '', content, flags=re.IGNORECASE).strip()
            q_blocks = re.split(r'\n\s*(?=\d+\.)', content)
            temp_list = []
            for block in q_blocks:
                clean_q = " ".join(block.strip().splitlines()).strip("0123456789. ").strip()
                if clean_q:
                    temp_list.append({"question": clean_q, "answer": ""})
            d["fill_in_the_blanks"] = temp_list

        if len(parts) > 3:
            content = parts[3]
            content = re.sub(r'.*subjective\s+questions.*?\n', '', content, flags=re.IGNORECASE).strip()
            q_blocks = re.split(r'\n\s*(?=\d+\.)', content)
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
async def generate_answers(
    questions_json: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...)
):
    ctx = get_context_from_selection(class_num, subject)
    try:
        q_data = QuestionsPayload.model_validate_json(questions_json).questions
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for questions: {e}")

    answered_data = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    for q_type, q_list in q_data.items():
        if not q_list:
            continue

        prompts = [f"{i+1}. {item.question}" for i, item in enumerate(q_list)]
        joined_prompts = "\n".join(prompts)
        
        p_template = ""
        if q_type == "mcqs":
            p_template = f"Context:\n{ctx}\n\nFor each question below, provide ONLY the correct option letter and text (e.g., 'C) Paris'), each on a new line.\n\n{joined_prompts}"
        elif q_type == "fill_in_the_blanks":
            p_template = f"Context:\n{ctx}\n\nFor each question below, provide ONLY the single word or short phrase that fits in the blank, each on a new line.\n\n{joined_prompts}"
        else:
            p_template = f"Context:\n{ctx}\n\nAnswer each of the following questions in detail. Start each answer with the corresponding number (e.g., '1. [Answer text]').\n\n{joined_prompts}"

        r = await a.generate(model="llama3.1", prompt=p_template)
        raw_answers = r["response"].strip()
        
        raw_answers = re.sub(r'^(.*?)(?=1\.)', '', raw_answers, flags=re.DOTALL)
        split_answers = re.split(r'\n\s*(?=\d+\.)', raw_answers)
        
        temp_list = []
        for i, item in enumerate(q_list):
            if i < len(split_answers) and split_answers[i].strip():
                answer_text = re.sub(r'^\d+\.\s*', '', split_answers[i].strip())
                item.answer = answer_text
            else:
                item.answer = "No answer generated."
            temp_list.append(item)
        
        answered_data[q_type] = temp_list
        
    return answered_data


@app.post("/evaluate-accuracy")
async def evaluate_accuracy(
    question: str = Form(...),
    generated_answer: str = Form(...),
    q_type: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...)
):
    ctx = get_context_from_selection(class_num, subject)

    if q_type == "subjective":
        emb1 = c.encode(ctx, convert_to_tensor=True)
        emb2 = c.encode(generated_answer, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2)
        score = sim.item()
        return {
            "generated_answer": generated_answer,
            "bert_answer": "N/A (Relevance Score)",
            "similarity_score": score
        }
    else:
        loop = asyncio.get_running_loop()
        bert_result = await loop.run_in_executor(
            None, b, question, ctx
        )
        bert_answer = bert_result["answer"]

        emb1 = c.encode(generated_answer, convert_to_tensor=True)
        emb2 = c.encode(bert_answer, convert_to_tensor=True)
        
        sim = util.pytorch_cos_sim(emb1, emb2)
        score = sim.item()
        return {
            "generated_answer": generated_answer,
            "bert_answer": bert_answer,
            "similarity_score": score
        }

@app.post("/regenerate-answer")
async def regenerate_answer(
    class_num: str = Form(...),
    subject: str = Form(...),
    question: str = Form(...),
    original_answer: str = Form(...),
    human_evaluation: str = Form(...),
    bert_answer: str = Form(...),
    q_type: str = Form(...)
):
    # For simple questions, the BERT answer is the best answer. Use it directly.
    if q_type in ["mcqs", "fill_in_the_blanks"]:
        return {"new_answer": bert_answer}

    # For subjective questions, we need the LLM to generate a better paragraph.
    elif q_type == "subjective":
        ctx = get_context_from_selection(class_num, subject)
        p = f"""
You are an expert teaching assistant tasked with correcting a detailed, subjective answer.
Your goal is to generate a new answer that is well-written, comprehensive, and factually correct based on the text.

**Full Context:**
---
{ctx}
---

**Question:**
{question}

**Previous Flawed Answer:**
"{original_answer}"

**Human Feedback:**
The previous answer was marked as: **{human_evaluation}**.

**Your Task:**
Generate a new, improved, and comprehensive answer to the subjective question. Base your new answer on the full context and ensure it is accurate and detailed.

**Output:**
Provide ONLY the new, corrected answer.
"""
        try:
            r = await a.generate(model="llama3.1", prompt=p)
            new_answer = r["response"].strip()
            if not new_answer:
                raise HTTPException(status_code=500, detail="Model failed to regenerate an answer.")
            return {"new_answer": new_answer}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Answer regeneration failed: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown question type for regeneration: {q_type}")

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