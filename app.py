## modules section
import asyncio
import json
import re
import os
import traceback
from io import BytesIO
from typing import Dict, List, Union, Callable

# FOR METRIC SYSTEM MODELS ARE IMPORTED HERE
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import ollama
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from openai import AsyncOpenAI
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuring Hugging Face models...")

# METRIC IS INITIALISED HERE 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Initialize ROUGE scorer globally
rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# MODELS HANDLING
a = None  # ollama 
f = None  # openai t
b = None  # QA pipeline
c = None  # SentenceTransformer
d = None  # summarizer pipeline
e = None  # T5 QG dict {tokenizer, model, device}
g = None  # gpt2 pipeline

# OLLAMA AND OPENAI MODEL
try:
    a = ollama.AsyncClient()
    print("Ollama client initialized.")
except Exception as ex:
    print(f"Error initializing Ollama client: {ex}")
    a = None

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in .env file. OpenAI models will not work.")
        f = None
    else:
        f = AsyncOpenAI(api_key=openai_api_key)
        print("OpenAI client (f) initialized.")
except Exception as ex:
    print(f"Error initializing OpenAI client: {ex}")
    f = None


use_cuda = torch.cuda.is_available()
hf_pipeline_device = 0 if use_cuda else -1    
st_device_str = "cuda" if use_cuda else "cpu"  

print(f"torch.cuda.is_available() = {use_cuda}. Using pipeline device={hf_pipeline_device}, sentence-transformer device={st_device_str}")


try:
    preferred_qa = "deepset/bart-base-squad2"
    fallback_qa = "distilbert-base-cased-distilled-squad"
    try:
        print(f"Loading preferred QA model: {preferred_qa}")
        b = pipeline("question-answering", model=preferred_qa, device=hf_pipeline_device)
        print(f"Loaded QA model: {preferred_qa}")
    except Exception as ex:
        print(f"Could not load preferred QA ({preferred_qa}): {ex}\nFalling back to {fallback_qa}")
        try:
            b = pipeline("question-answering", model=fallback_qa, device=hf_pipeline_device)
            print(f"Loaded fallback QA model: {fallback_qa}")
        except Exception as ex2:
            print(f"Failed to load fallback QA model ({fallback_qa}): {ex2}")
            b = None

    # SENTENCE TRANSFORMER 
    try:
        print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        c = SentenceTransformer("all-MiniLM-L6-v2", device=st_device_str)
        print("SentenceTransformer loaded.")
    except Exception as ex:
        print(f"Failed to load SentenceTransformer: {ex}")
        traceback.print_exc()
        c = None

    # BART-MODEL
    preferred_sum = "facebook/bart-large-cnn"
    fallback_sum = "sshleifer/distilbart-cnn-12-6"
    try:
        print(f"Loading summarizer: {preferred_sum}")
        d = pipeline("summarization", model=preferred_sum, device=hf_pipeline_device)
        print(f"Loaded summarizer: {preferred_sum}")
    except Exception as ex:
        print(f"Could not load preferred summarizer ({preferred_sum}): {ex}\nFalling back to {fallback_sum}")
        try:
            d = pipeline("summarization", model=fallback_sum, device=hf_pipeline_device)
            print(f"Loaded fallback summarizer: {fallback_sum}")
        except Exception as ex2:
            print(f"Failed to load fallback summarizer ({fallback_sum}): {ex2}")
            d = None

    # T5 GENERATOR
    preferred_t5 = "voidful/t5-base-qg-hl"
    fallback_t5 = "valhalla/t5-small-qg-hl"
    try:
        print(f"Attempting T5 QG model: {preferred_t5}")
        e_tokenizer = T5Tokenizer.from_pretrained(preferred_t5)
        e_model = T5ForConditionalGeneration.from_pretrained(preferred_t5)
        e = {"tokenizer": e_tokenizer, "model": e_model, "device": st_device_str}
        if use_cuda:
            e_model.to("cuda")
        else:
            e_model.to("cpu")
        print(f"Loaded T5 QG: {preferred_t5}")
    except Exception as ex:
        print(f"Preferred T5 failed: {ex}\nTrying fallback T5: {fallback_t5}")
        try:
            e_tokenizer = T5Tokenizer.from_pretrained(fallback_t5)
            e_model = T5ForConditionalGeneration.from_pretrained(fallback_t5)
            e = {"tokenizer": e_tokenizer, "model": e_model, "device": st_device_str}
            if use_cuda:
                e_model.to("cuda")
            else:
                e_model.to("cpu")
            print(f"Loaded fallback T5 QG: {fallback_t5}")
        except Exception as ex2:
            print(f"Failed to load any T5 QG model: {ex2}")
            traceback.print_exc()
            e = None

    # GPT-2 QUESTION GENERATOR
    try:
        print("Loading GPT-2 generator (gpt2)...")
        g = pipeline("text-generation", model="gpt2", device=hf_pipeline_device)
        print("GPT-2 generator loaded.")
    except Exception as ex:
        print(f"Failed to load GPT-2 generator: {ex}")
        traceback.print_exc()
        g = None

    print("Model loading complete.")
except Exception:
    print("Unexpected error during model initialization:")
    traceback.print_exc()
    b = None
    c = None
    d = None
    e = None
    g = None

# FILE PATHS
BOOKS_BASE_DIR = "BOOKS"
if not os.path.exists(BOOKS_BASE_DIR):
    try:
        os.makedirs(BOOKS_BASE_DIR)
        print(f"Created base directory: {BOOKS_BASE_DIR}")
    except OSError as ex:
        print(f"Error creating directory {BOOKS_BASE_DIR}: {ex}")
        BOOKS_BASE_DIR = "."

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

    @field_validator('mcqs', 'fill_in_the_blanks', 'subjective')
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Question count must be non-negative')
        return v

class QuestionItem(BaseModel):
    question: str
    answer: Union[str, None] = None

class QuestionsPayload(BaseModel):
    questions: Dict[str, List[QuestionItem]]

class PdfRequest(BaseModel):
    answered_questions: Dict[str, List[QuestionItem]]

class QuestionEvaluationRequest(BaseModel):
    question: str
    class_num: str
    subject: str

class HumanEvaluationLog(BaseModel):
    question: str
    q_type: str
    evaluation: str

class QuestionRating(BaseModel):
    difficulty: str = ""
    remarks: str = ""

class InvigilatorRatingData(BaseModel):
    ratings: Dict[str, Dict[str, QuestionRating]] = {}
    overall_remarks: str = ""

class DifficultyRatings(BaseModel):
    class_num: str
    subject: str
    invigilator_data: Dict[str, InvigilatorRatingData]

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
            if filename.lower().startswith(file_prefix.lower()) and filename.lower().endswith(".txt"):
                found_file_path = os.path.join(class_dir, filename)
                break
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Base directory '{BOOKS_BASE_DIR}' not found.")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error accessing class directory: {ex}")
    if not found_file_path:
        raise HTTPException(status_code=404, detail=f"Textbook file for Class {class_num}, Subject {subject} not found.")
    try:
        with open(found_file_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
            return content
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error reading file '{os.path.basename(found_file_path)}': {ex}")

def run_t5_qg(context: str, model_dict: dict):
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    sentences = re.split(r'[.!?] ', context)
    target_sentences = [s for s in sentences if len(s) > 50]
    if not target_sentences:
        target_sentences = sentences
    if not target_sentences:
        return "Could not find a sentence to use for question generation."
    target_sentence = target_sentences[len(target_sentences) // 2]
    highlighted_context = context.replace(target_sentence, f"<hl> {target_sentence} <hl>")
    input_text = f"generate question: {highlighted_context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=64)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question.replace("question: ", "").strip()

def parse_questions_from_block(content_block: str) -> List[Dict[str, Union[str, None]]]:
    q_list = []
    if not content_block:
        return q_list
    content_block = re.sub(r'(?:^|\n)\s*(?:\*\*)?Part \d.*?(?:\n|$)', '', content_block, flags=re.IGNORECASE)
    q_blocks = re.split(r'\n\s*(?=\d+\.)', content_block.strip())
    for block in q_blocks:
        if not block:
            continue
        if re.match(r'^\s*(?:\*\*)?Part', block, re.IGNORECASE):
            continue
        parts = re.split(r'\n\s*Answer:.*', block, flags=re.IGNORECASE | re.DOTALL)
        question_text = parts[0]
        clean_q = re.sub(r'^\d+\.\s*', '', question_text.strip()).strip()
        clean_q = re.sub(r'^\s*Here.*?:\s*\n?', '', clean_q, flags=re.IGNORECASE | re.DOTALL).strip()
        if "**Options:**" in clean_q:
            clean_q = clean_q.split("**Options:**")[0]
        if "Part " in clean_q and len(clean_q) < 20:
            continue
        if clean_q:
            q_list.append({"question": clean_q, "answer": None})
    return q_list


def sanitize_chunk_output(raw: str) -> str:
    """
    Remove wrapper/debug noise from model output and try to extract the first
    numbered list block. Returns cleaned text.
    """
    if not raw:
        return ""
    t = raw

   
    m = re.search(r'response\s*=\s*"(.*?)"(?:\s|$)', t, flags=re.DOTALL)
    if m:
        t = m.group(1)

    t = re.sub(r'\b(?:model|created_at|done|done_reason|total_duration|load_duration|prompt_eval_count|prompt_eval_duration|eval_count|eval_duration|thinking|context|response)\s*=\s*(?:\[[^\]]*\]|"[^"]*"|\'[^\']*\'|[^\s,;]+)', ' ', t)

    t = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', ' ', t)

    t = re.sub(r'(?i)prompt[:=]\s*[^\\n]+', ' ', t)

    m2 = re.search(r'(?:^|\n)((?:\s*\d+\.\s.*(?:\n|$))+)', t)
    if m2:
        t2 = m2.group(1)
        lines = t2.strip().splitlines()
        filtered = [ln for ln in lines if re.match(r'^\s*\d+\.', ln)]
        if filtered:
            return "\n".join(filtered).strip()

    m3 = re.search(r'(Here are .*?questions:|Here are the .*?MCQs based on the provided context:)(.*)', t, flags=re.IGNORECASE|re.DOTALL)
    if m3:
        body = m3.group(2)
        m4 = re.search(r'((?:\s*\d+\.\s.*(?:\n|$))+)', body)
        if m4:
            return m4.group(1).strip()
        return re.sub(r'\s+', ' ', body).strip()

    t = re.sub(r'\b\w+:\s*\S+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    return t.strip()

def chunk_text_for_gpt2(text: str, approx_chunk_chars: int = 1800) -> List[str]:
    """
    Heuristic chunking: split `text` into chunks around sentence boundaries
    so that each chunk has roughly <= approx_chunk_chars characters.
    """
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = []
    curr_len = 0
    for s in sentences:
        slen = len(s) + 1
        if curr_len + slen > approx_chunk_chars and current:
            chunks.append(" ".join(current).strip())
            current = [s]
            curr_len = slen
        else:
            current.append(s)
            curr_len += slen
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

async def _run_llm_call(model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """
    Unified call handler:
    - OpenAI chat models: uses OpenAI async client f
    - gpt2 (local HF pipeline): uses run_in_executor with lambda wrapper (no unexpected kwargs)
    - Ollama: uses async client a
    - T5: Uses local pipeline e
    """
    t = ""
    if model_name.startswith("gpt-"):
        if not f:
            raise HTTPException(status_code=503, detail="OpenAI client (f) is not available.")
        response = await f.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful quiz-generation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        t = response.choices[0].message.content.strip()
    elif model_name == "t5":
        if not e:
            raise HTTPException(status_code=503, detail="T5 model (e) is not available.")
        loop = asyncio.get_running_loop()
        tokenizer = e['tokenizer']
        model = e['model']
        device = e['device']
        def _t5_gen():
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_length=max_tokens)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        t = await loop.run_in_executor(None, _t5_gen)
    elif model_name == "gpt2":
        if not g:
            raise HTTPException(status_code=503, detail="GPT-2 client (g) is not available.")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: g(
                    prompt,
                    max_length=min(max_tokens + len(prompt.split()), 1024),
                    do_sample=True,
                    temperature=0.7,
                    truncation=True,
                    pad_token_id=50256,
                    num_return_sequences=1
                )
            )
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                full_text = result.get('generated_text', '')
            else:
                full_text = str(result)
            if full_text.startswith(prompt):
                t = full_text[len(prompt):].strip()
            else:
                t = full_text.strip()
        except TypeError:
            result = await loop.run_in_executor(None, lambda: g(prompt, max_length=512, do_sample=True, pad_token_id=50256))
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                full_text = result.get('generated_text', '')
            else:
                full_text = str(result)
            if full_text.startswith(prompt):
                t = full_text[len(prompt):].strip()
            else:
                t = full_text.strip()
    else:
        if not a:
            raise HTTPException(status_code=503, detail="Ollama client (a) is not available.")
        response_data = await a.generate(model=model_name, prompt=prompt, stream=False)
        t = response_data.get("response", "").strip()
    return t

async def _generate_fallback_questions(
    ctx: str,
    q_type: str,
    count: int,
    model_name: str,
    parse_func: Callable[[str], List[Dict[str, Union[str, None]]]]
) -> List[Dict[str, Union[str, None]]]:
    prompt_instruction = ""
    if q_type == "mcqs":
        prompt_instruction = f"Generate exactly {count} Multiple Choice Questions (MCQs) with four options (A, B, C, D). Do not include the answer."
    elif q_type == "fill_in_the_blanks":
        prompt_instruction = f"Generate exactly {count} fill-in-the-blanks questions. Replace a key word or phrase with '____'. Do not include the answer."
    elif q_type == "subjective":
        prompt_instruction = f"Generate exactly {count} subjective (long-answer) questions. Do not include the answer."
    else:
        return []

    p = f"""
You are an assistant that generates a specific type of question from the provided context.
Base all questions ONLY on the "Full Context" provided below.
**Full Context:**
---
{ctx}
---
**Task:**
{prompt_instruction}
**Output:**
Provide ONLY the numbered list of questions. Do not include any other text, headers, or explanations.
"""
    try:
        t = await _run_llm_call(model_name, p, max_tokens=(count * 150))
        if not t:
            return []
        return parse_func(t)
    except Exception as ex:
        traceback.print_exc()
        return []

# API END-POINTS
@app.post("/generate-questions")
async def generate_questions(
    class_num: str = Form(...),
    subject: str = Form(...),
    counts_json: str = Form(...),
    model_name: str = Form("llama3.1")
):
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        raise ex
    try:
        counts = QuestionCounts.model_validate_json(counts_json)
    except ValidationError as ex:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for question counts: {ex}")

    d_out = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    try:
        # T5 REFINED TO PREVENT HALLUCINATIONS
        if model_name == "t5":
            sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', ctx.strip()) if len(s.strip()) > 30]
            if not sentences: sentences = [ctx]

            async def gen_t5_smart(sentence):
                words = sentence.split()
                keywords = [w for w in words if len(w) > 4 and w.lower() not in ("which", "where", "about", "their", "there")]
                target = keywords[0] if keywords else (words[-1] if words else "it")
                
                p = f"generate question: <hl> {target} <hl> {sentence}"
                return await _run_llm_call("t5", p, max_tokens=64)

            # MCQ QUESTIONS
            for i in range(counts.mcqs):
                if not sentences: break
                s_idx = i % len(sentences)
                q = await gen_t5_smart(sentences[s_idx])
                dist1 = sentences[(s_idx + 1) % len(sentences)].split()[0] + "..."
                dist2 = sentences[(s_idx + 2) % len(sentences)].split()[0] + "..."
                d_out["mcqs"].append({"question": f"{q}\nA) {dist1}\nB) {dist2}\nC) Correct Choice\nD) None", "answer": None})

            # Fill in blanks
            for i in range(counts.fill_in_the_blanks):
                if not sentences: break
                s_idx = (i + 5) % len(sentences)
                q = await gen_t5_smart(sentences[s_idx])
                fib_q = q.replace("What", "____").replace("Which", "____").replace("?", ".")
                d_out["fill_in_the_blanks"].append({"question": fib_q, "answer": None})

        # Subjectve QUESTIONS
            for i in range(counts.subjective):
                if not sentences: break
                s_idx = (i + 3) % len(sentences)
                q = await gen_t5_smart(sentences[s_idx])
                d_out["subjective"].append({"question": q, "answer": None})

            return {"questions": d_out, "context": ctx}

        elif model_name == "gpt2":
            chunks = chunk_text_for_gpt2(ctx, approx_chunk_chars=1800)
            if not chunks: chunks = [ctx[:4000]]
            n_chunks = len(chunks)
            mcq_alloc = [counts.mcqs // n_chunks] * n_chunks
            fib_alloc = [counts.fill_in_the_blanks // n_chunks] * n_chunks
            subj_alloc = [counts.subjective // n_chunks] * n_chunks
            for i in range(counts.mcqs % n_chunks): mcq_alloc[i] += 1
            for i in range(counts.fill_in_the_blanks % n_chunks): fib_alloc[i] += 1
            for i in range(counts.subjective % n_chunks): subj_alloc[i] += 1

            agg_mcq_text, agg_fib_text, agg_subj_text = "", "", ""
            for idx, chunk in enumerate(chunks):
                sub_prompts = []
                if mcq_alloc[idx] > 0: sub_prompts.append(f"Generate {mcq_alloc[idx]} MCQs. List 1. ...")
                if fib_alloc[idx] > 0: sub_prompts.append(f"Generate {fib_alloc[idx]} blanks.")
                if subj_alloc[idx] > 0: sub_prompts.append(f"Generate {subj_alloc[idx]} subjective questions.")
                if not sub_prompts: continue
                prompt_chunk = "\n".join(sub_prompts) + "\n" + chunk
                raw_out = await _run_llm_call("gpt2", prompt_chunk, max_tokens=512)
                cleaned = sanitize_chunk_output(raw_out)
                agg_mcq_text += "\n" + cleaned
                agg_fib_text += "\n" + cleaned
                agg_subj_text += "\n" + cleaned

            d_out["mcqs"] = parse_questions_from_block(agg_mcq_text)
            d_out["fill_in_the_blanks"] = parse_questions_from_block(agg_fib_text)
            d_out["subjective"] = parse_questions_from_block(agg_subj_text)
            
            if not d_out["mcqs"] and counts.mcqs > 0:
                d_out["mcqs"] = await _generate_fallback_questions(ctx, "mcqs", counts.mcqs, "gpt2", parse_questions_from_block)
            if not d_out["fill_in_the_blanks"] and counts.fill_in_the_blanks > 0:
                d_out["fill_in_the_blanks"] = await _generate_fallback_questions(ctx, "fill_in_the_blanks", counts.fill_in_the_blanks, "gpt2", parse_questions_from_block)
            if not d_out["subjective"] and counts.subjective > 0:
                d_out["subjective"] = await _generate_fallback_questions(ctx, "subjective", counts.subjective, "gpt2", parse_questions_from_block)

            return {"questions": d_out, "context": ctx}

        else:
            #llama LOGIC
            p = f"""
You are an assistant that generates a three-part quiz based on the provided text.
Base all questions ONLY on the "Full Context" provided below.
**Full Context:**
---
{ctx}
---
Follow these instructions exactly.
**Part 1: Multiple Choice Questions**
Generate exactly {counts.mcqs} MCQs. **Do NOT include the answer.**

**Part 2: Fill in the Blanks**
Generate exactly {counts.fill_in_the_blanks} fill-in-the-blanks questions. **Do NOT include the answer.**

**Part 3: Subjective Questions**
Generate exactly {counts.subjective} subjective (long-answer) questions. **Do NOT include the answer.**
"""
            t = await _run_llm_call(model_name, p, max_tokens=2048)

            def get_part_start_index(text, part_num):
                pattern = r'(?:^|\n)\s*(?:\*\*)?\s*Part\s*' + str(part_num)
                match = re.search(pattern, text, re.IGNORECASE)
                return match.start() if match else -1

            idx_1 = get_part_start_index(t, 1)
            idx_2 = get_part_start_index(t, 2)
            idx_3 = get_part_start_index(t, 3)
            total_len = len(t)
            mcq_content = t[idx_1:(idx_2 if idx_2 != -1 else (idx_3 if idx_3 != -1 else total_len))] if idx_1 != -1 else ""
            fib_content = t[idx_2:(idx_3 if idx_3 != -1 else total_len)] if idx_2 != -1 else ""
            sub_content = t[idx_3:] if idx_3 != -1 else ""

            d_out["mcqs"] = parse_questions_from_block(mcq_content)
            d_out["fill_in_the_blanks"] = parse_questions_from_block(fib_content)
            d_out["subjective"] = parse_questions_from_block(sub_content)

            if not d_out["mcqs"] and counts.mcqs > 0:
                d_out["mcqs"] = await _generate_fallback_questions(ctx, "mcqs", counts.mcqs, model_name, parse_questions_from_block)
            if not d_out["fill_in_the_blanks"] and counts.fill_in_the_blanks > 0:
                d_out["fill_in_the_blanks"] = await _generate_fallback_questions(ctx, "fill_in_the_blanks", counts.fill_in_the_blanks, model_name, parse_questions_from_block)
            if not d_out["subjective"] and counts.subjective > 0:
                d_out["subjective"] = await _generate_fallback_questions(ctx, "subjective", counts.subjective, model_name, parse_questions_from_block)

            return {"questions": d_out, "context": ctx}

    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question generation failed: {ex}")

# BART MODEL USED, TO GENERATE QUESTIONS
@app.post("/generate-bart-question")
async def generate_bart_question(
    class_num: str = Form(...),
    subject: str = Form(...),
    q_type: str = Form(...)
):
    if not e:
        raise HTTPException(status_code=503, detail="T5/BART Question Generator (e) not available.")
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        raise ex
    loop = asyncio.get_running_loop()
    new_q = await loop.run_in_executor(None, run_t5_qg, ctx[:4000], e)
    if not new_q:
        raise HTTPException(status_code=500, detail="Model 'e' returned an empty question.")
    return {"new_question": new_q}

# FOR ANSWER GENERATION
@app.post("/generate-answers")
async def generate_answers(
    questions_json: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...),
    model_name: str = Form("llama3.1")
):
    """
    Robust generate_answers.
    FIX: If T5 generated the questions, use Llama/GPT to answer them.
    """
    if not a and not f and not g:
        raise HTTPException(status_code=503, detail="No generation clients available.")

    print(f"generate_answers called for class={class_num}, subject={subject}, requested_model={model_name}")
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        print(f"Unexpected error getting context for answers: {ex}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load context data.")

    if not ctx:
        print("Error: Cannot generate answers without context.")
        raise HTTPException(status_code=400, detail="Context file is empty or unreadable.")

    try:
        payload = QuestionsPayload.model_validate_json(questions_json)
        q_data = payload.questions
    except ValidationError as ex:
        print(f"Error validating input questions JSON: {ex}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for questions: {ex}")

    answered_data = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    answer_model = model_name
    if model_name == "t5":
        if a:
            answer_model = "llama3.1" 
        elif f:
            answer_model = "gpt-3.5-turbo"
        else:
            answer_model = "gpt2" 

    fallback_order = [answer_model]
    if "gpt2" not in fallback_order:
        fallback_order.append("gpt2")
    if f and "gpt-3.5-turbo" not in fallback_order:
        fallback_order.append("gpt-3.5-turbo")

    async def try_generate(prompt: str, max_tokens: int = 1024):
        last_exc = None
        for mname in fallback_order:
            try:
                print(f"Attempting generation with model: {mname}")
                out = await _run_llm_call(mname, prompt, max_tokens=max_tokens)
                if out and len(out.strip()) > 0:
                    print(f"Model {mname} returned a response (len={len(out.strip())})")
                    return out
                else:
                    print(f"Model {mname} returned empty string.")
            except Exception as ex:
                last_exc = ex
                print(f"Model {mname} generation failed: {ex}")
                traceback.print_exc()
                continue
        raise Exception(f"All model attempts failed. Last error: {last_exc}")

    # FUNCTION USED TO EXTRACT OPTIONS FOR MCQ 
    def extract_options_from_question(qtext: str):
        """
        Returns dict mapping letter->option_text, e.g. {'a': 'One hour', 'b': 'Two hours', ...}
        """
        opts = {}
        lines = qtext.splitlines()
        combined = "\n".join(lines)
        for match in re.finditer(r'([A-Da-d])\s*[\)\.\:]\s*(.+?)(?=(?:\n[A-Da-d]\s*[\)\.\:]|$))', combined, flags=re.DOTALL):
            letter = match.group(1).lower()
            text = match.group(2).strip()
            text = re.sub(r'\s+', ' ', text).strip()
            opts[letter] = text
        if not opts:
            inline_matches = re.findall(r'([A-Da-d])\s*[\)\.\:]\s*([^A-Da-d]+)', combined)
            for lm in inline_matches:
                letter = lm[0].lower()
                text = lm[1].strip()
                text = re.sub(r'\s+', ' ', text).strip()
                if letter and text:
                    opts[letter] = text
        return opts

    for q_type, q_list in q_data.items():
        if not q_list:
            print(f"Skipping empty question type: {q_type}")
            continue

        prompts = []
        original_questions = []
        for i, item in enumerate(q_list):
            original_questions.append(item.question)
            clean_q = item.question
            if isinstance(clean_q, str):
                if "**Options:**" in clean_q:
                    clean_q = clean_q.split("**Options:**")[0]
                elif "\nA)" in clean_q:
                    clean_q = clean_q.split("\nA)")[0]
                clean_q = re.sub(r'^(?:\*\*)?Part \d.*?(?:\*\*)?\n?', '', clean_q, flags=re.IGNORECASE)
                prefixes_patterns = [ r"^Here.*?text:", r"^Here is a new question:", r"^\*\*Question:\*\*", r"^\d+\.\s*" ]
                for pattern in prefixes_patterns:
                    clean_q = re.sub(pattern, "", clean_q, flags=re.IGNORECASE).strip()
                clean_q = clean_q.replace("**", "").strip()
            else:
                clean_q = "Invalid question format"
            prompts.append(f"{i+1}. {clean_q}")

        joined_prompts = "\n".join(prompts)

        if q_type == "mcqs":
            p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: For each numbered question, provide ONLY the correct option letter and text (e.g., '1. C) Paris' or '1. C) Paris, France'). DO NOT repeat the question text. Format: 'Number. Option'"
        elif q_type == "fill_in_the_blanks":
            p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: For each numbered question, provide ONLY the single missing word or short phrase. DO NOT repeat the question. Format: '1. Answer'"
        else:
            p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: Answer each numbered question in detail based on the context. Start with the number. DO NOT repeat the question text."

        raw_answers = ""
        try:
            raw_answers = await try_generate(p_template, max_tokens=1024)
            raw_answers = re.sub(r'```[\w\s]*\n?', '', raw_answers)
            raw_answers = re.sub(r'\n?```', '', raw_answers)
            raw_answers = re.sub(r'^\*+\s*|\s*\*+$', '', raw_answers).strip()
            raw_answers = re.sub(r'^(.*?)(?=1[\.\)])', '', raw_answers, flags=re.DOTALL | re.IGNORECASE).strip()

            split_answers = re.split(r'\n\s*(?=\d+[\.\)])', raw_answers)
            parsed_answers = {}
            for ans_block in split_answers:
                match = re.match(r'(\d+)[\.\)]\s*(.*)', ans_block.strip(), re.DOTALL)
                if match:
                    q_num = int(match.group(1))
                    ans_text = match.group(2).strip()
                    ans_text = re.sub(r'^\*+\s*|\s*\*+$', '', ans_text).strip()
                    if ans_text.lower().startswith(("what", "why", "how")) and "?" in ans_text:
                        parts = ans_text.split("?")
                        if len(parts) > 1:
                            ans_text = parts[-1].strip()
                    parsed_answers[q_num - 1] = ans_text

            temp_list = []
            for i, original_q_text in enumerate(original_questions):
                ans_raw = parsed_answers.get(i, None)
                if not ans_raw or not str(ans_raw).strip():
                    final_answer = "No answer generated or parsing failed."
                else:
                    if q_type == "mcqs":
                        candidate = ans_raw.strip()
                        m_letter = re.match(r'^\s*([A-Da-d])\s*[\)\.\:]*\s*(.*)$', candidate)
                        if m_letter:
                            letter = m_letter.group(1).lower()
                            remainder = m_letter.group(2).strip()
                            if remainder:
                                final_answer = f"{letter}) {remainder}"
                            else:
                                opts = extract_options_from_question(original_q_text or "")
                                opt_text = opts.get(letter)
                                if opt_text:
                                    final_answer = f"{letter}) {opt_text}"
                                else:
                                    final_answer = f"{letter}) (option text not found)"
                        else:
                            final_answer = ans_raw
                    else:
                        final_answer = ans_raw

                temp_list.append(QuestionItem(question=original_q_text, answer=final_answer))
            answered_data[q_type] = temp_list

        except Exception as ex:
            print(f"Failed to generate answers for {q_type}: {ex}")
            traceback.print_exc()
            temp_list = []
            for q in original_questions:
                temp_list.append(QuestionItem(question=q, answer="Error generating answer: " + str(ex)))
            answered_data[q_type] = temp_list

    print("Finished generating all answers.")
    return answered_data

# ACCURACY EVALUATION
@app.post("/evaluate-accuracy")
async def evaluate_accuracy(
    question: str = Form(...),
    generated_answer: str = Form(...),
    q_type: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...),

    # BERT, BLEU, ROUGE USED HERE

    metric: str = Form("bert") \
):
    print(f"evaluate_accuracy called: q_type={q_type}, metric={metric}, question[:80]={question[:80]!r}")
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        return {"generated_answer": generated_answer, "bert_answer": f"Error loading context: {ex.detail}", "similarity_score": 0.0}
    except Exception as ex:
        traceback.print_exc()
        return {"generated_answer": generated_answer, "bert_answer": "Unexpected error loading context.", "similarity_score": 0.0}
    if not ctx:
        return {"generated_answer": generated_answer, "bert_answer": "Error: Context unavailable", "similarity_score": 0.0}

    if b and c:
        loop = asyncio.get_running_loop()
        try:
            reference_text = ""
            display_ref_answer = ""
            if q_type == "subjective":
                reference_text = ctx
                display_ref_answer = "N/A (Relevance Score)"
            else:
                bert_result = await loop.run_in_executor(None, lambda: b(question=question, context=ctx))
                if isinstance(bert_result, list) and len(bert_result) > 0:
                    bert_item = bert_result[0]
                elif isinstance(bert_result, dict):
                    bert_item = bert_result
                else:
                    bert_item = {}
                reference_text = bert_item.get("answer", "")
                if not reference_text or reference_text.startswith(("BART could", "Error")):
                    return {"generated_answer": generated_answer, "bert_answer": "BART failed to extract ref", "similarity_score": 0.0}
                display_ref_answer = reference_text

            gen = generated_answer.strip()
            gen = re.sub(r'^\d+[\.\)]\s*', '', gen)
            m = re.match(r'^[A-Da-d][\)\.\:]\s*(.*)', gen)
            gen_text = m.group(1).strip() if m else gen
            score = 0.0
            
            #BLEU MODEL
            if metric == "bleu":
                ref_tokens = [nltk.word_tokenize(reference_text)]
                cand_tokens = nltk.word_tokenize(gen_text)
                score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method1)

            #ROUGE MODEL
            elif metric == "rouge":
                scores = rouge_evaluator.score(reference_text, gen_text)
                score = scores['rougeL'].fmeasure

            #BERT MODEL
            else:
                emb_ref = c.encode(reference_text, convert_to_tensor=True)
                emb_gen = c.encode(gen_text, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(emb_ref, emb_gen)
                score = float(sim.item())

            return {"generated_answer": generated_answer, "bert_answer": display_ref_answer, "similarity_score": score}

        except Exception as ex:
            print("Error during full-model evaluation. Falling back to heuristic.")
            traceback.print_exc()

    print("Using fallback heuristic evaluation (models unavailable).")
    sentences = re.split(r'(?<=[\.\?\!])\s+', ctx.strip())
    def str_sim(a: str, b: str) -> float:
        try:
            a_norm = re.sub(r'\s+', ' ', a.strip().lower())
            b_norm = re.sub(r'\s+', ' ', b.strip().lower())
            if not a_norm or not b_norm:
                return 0.0
            return SequenceMatcher(None, a_norm, b_norm).ratio()
        except Exception:
            return 0.0

    if q_type in ("mcqs", "fill_in_the_blanks"):
        gen = generated_answer.strip()
        gen = re.sub(r'^\d+[\.\)]\s*', '', gen)
        m = re.match(r'^[A-Da-d][\)\.\:]\s*(.*)', gen)
        gen_text = m.group(1).strip() if m else gen
        max_sim = 0.0
        for s in sentences:
            s_small = s.strip()
            if not s_small:
                continue
            sim = str_sim(gen_text, s_small)
            if sim > max_sim:
                max_sim = sim
        if gen_text and gen_text.lower() in ctx.lower():
            max_sim = max(max_sim, 1.0)
        score = float(max_sim)
        return {"generated_answer": generated_answer, "bert_answer": "N/A (Fallback)", "similarity_score": score}
    else:
        stopwords = {"the","is","and","a","an","in","on","for","of","to","was","were","it","that","this","these","those","are","by","with","as","at","from"}
        def word_set(text: str):
            words = re.findall(r'\w+', text.lower())
            return set(w for w in words if w not in stopwords)
        ctx_set = word_set(ctx)
        gen_set = word_set(generated_answer)
        if not gen_set or not ctx_set:
            score = 0.0
        else:
            inter = ctx_set.intersection(gen_set)
            avg_len = (len(ctx_set) + len(gen_set)) / 2.0
            score = float(len(inter) / avg_len) if avg_len > 0 else 0.0
            score = max(0.0, min(1.0, score))
        return {"generated_answer": generated_answer, "bert_answer": "N/A (Fallback)", "similarity_score": score}

@app.post("/answer-questions-pdf")
async def answer_questions_pdf(req: PdfRequest):
    data = req.answered_questions
    if not data or not any(data.values()):
        raise HTTPException(status_code=400, detail="No answered questions provided.")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    try:
        story.append(Paragraph("Generated Answers", styles['h1']))
        story.append(Spacer(1, 24))
        for q_type, q_list in data.items():
            if q_list:
                header = q_type.replace('_', ' ').title()
                story.append(Paragraph(header, styles['h2']))
                for i, item in enumerate(q_list, 1):
                    q_text_raw = str(item.question) if item.question else "No Question Text Provided"
                    a_text_raw = str(item.answer) if item.answer else "No Answer Provided"
                    q_text_safe = q_text_raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    a_text_safe = a_text_raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    q_text_pdf = f"{i}. <b>{q_text_safe}</b>".replace('\n', '<br/>')
                    story.append(Paragraph(q_text_pdf, styles['Normal']))
                    story.append(Spacer(1, 6))
                    a_text_pdf = a_text_safe.replace('\n', '<br/>')
                    story.append(Paragraph(a_text_pdf, styles['Normal']))
                    story.append(Spacer(1, 12))
        doc.build(story)
        buffer.seek(0)
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {ex}")
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=generated_answers.pdf"})

@app.post("/evaluate-question-bert")
async def evaluate_question_bert(req: QuestionEvaluationRequest):
    if not b:
        raise HTTPException(status_code=503, detail="QA model (BART) not available.")
    try:
        ctx = get_context_from_selection(req.class_num, req.subject)
    except HTTPException as ex:
        return {"question": req.question, "bert_confidence_score": 0.0, "inferred_answer": f"Error loading context: {ex.detail}"}
    loop = asyncio.get_running_loop()
    bert_result = await loop.run_in_executor(None, lambda: b(question=req.question, context=ctx))
    if isinstance(bert_result, list) and len(bert_result) > 0:
        bert_item = bert_result[0]
    elif isinstance(bert_result, dict):
        bert_item = bert_result
    else:
        bert_item = {}
    confidence_score = bert_item.get("score", 0.0)
    found_answer = bert_item.get("answer", "BART could not find an answer.")
    return {"question": req.question, "bert_confidence_score": confidence_score, "inferred_answer": found_answer}

@app.post("/log-human-evaluation")
async def log_human_evaluation(log: HumanEvaluationLog):
    print(f"Human Feedback Received: q_type={log.q_type}, evaluation={log.evaluation}")
    return {"status": "success", "message": "Human evaluation logged."}

@app.post("/regenerate-question")
async def regenerate_question(
    class_num: str = Form(...),
    subject: str = Form(...),
    q_type: str = Form(...),
    old_question: str = Form(None),
    human_evaluation: str = Form(None),
    model_name: str = Form("llama3.1")
):
    if not d:
        raise HTTPException(status_code=503, detail="Summarizer model (d) not available.")
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        raise ex
    loop = asyncio.get_running_loop()
    try:
        context_chunk = ctx[:4000]
        summary_result = await loop.run_in_executor(None, lambda: d(context_chunk, max_length=150, min_length=30, do_sample=False))
        summary_text = summary_result[0]['summary_text']
        key_content = summary_text.split('.')[0] + '.'
    except Exception:
        traceback.print_exc()
        sentences = ctx.split('.')
        fallback_index = min(5, len(sentences) - 1)
        key_content = sentences[fallback_index].strip() + '.'
    if q_type == "mcqs":
        prompt_task = f'You are a question generator. Turn the following fact into one MCQ with four options. Fact: "{key_content}"'
    elif q_type == "fill_in_the_blanks":
        prompt_task = f'You are a question generator. Turn the following fact into one fill-in-the-blanks question. Fact: "{key_content}"'
    else:
        prompt_task = f'You are a question generator. Turn the following fact into one subjective question. Fact: "{key_content}"'
    new_q = await _run_llm_call(model_name, prompt_task, max_tokens=256)
    new_q = re.sub(r'```[\w\s]*\n?', '', new_q)
    new_q = re.sub(r'\n?```', '', new_q)
    new_q = new_q.strip()
    if not new_q:
        raise HTTPException(status_code=500, detail="Model failed to generate a question.")
    return {"new_question": new_q}

@app.get("/")
async def root():
    return {
        "message": "Quiz Generator API is running",
        "ollama_client_status": "Initialized" if a else "Failed",
        "openai_client_status": "Initialized" if f else "Failed (Check API Key)",
        "bart_qa_status": "Loaded" if b else "Failed",
        "sentence_transformer_status": "Loaded" if c else "Failed",
        "bart_summarizer_status": "Loaded" if d else "Failed",
        "t5_qg_status": "Loaded" if e else "Failed",
        "gpt2_gen_status": "Loaded" if g else "Failed"
    }

@app.get("/check-models")
async def check_models():
    ollama_available = False
    models_list = []
    if a:
        try:
            list_response = await a.list()
            ollama_available = True
            models_list = [m['name'] for m in list_response.get('models', [])]
        except Exception as ex:
            print(f"Could not reach Ollama service: {ex}")
            ollama_available = False
    return {
        "ollama_client": "Initialized" if a else "Failed",
        "ollama_service_reachable": ollama_available,
        "ollama_models_available": models_list,
        "openai_client": "Initialized" if f else "Failed (Check API Key)",
        "qa_model_status": "Loaded (QA)" if b else "Failed",
        "sentence_transformer": "Loaded" if c else "Failed",
        "bart_summarizer": "Loaded" if d else "Failed",
        "t5_qg_model": "Loaded" if e else "Failed",
        "gpt2_gen_model": "Loaded" if g else "Failed"
    }

@app.post("/submit-difficulty-ratings")
async def submit_difficulty_ratings(ratings_data: DifficultyRatings):
    print("--- Received Difficulty Ratings ---")
    print(f"Class: {ratings_data.class_num}, Subject: {ratings_data.subject}")
    print(f"Ratings Data: {ratings_data.model_dump_json(indent=2)}")
    return {"status": "success", "message": "Ratings logged."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
