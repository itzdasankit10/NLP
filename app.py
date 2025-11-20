import asyncio
import json
import re
import os
from io import BytesIO
from typing import Dict, List, Union, Callable

import ollama
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sentence_transformers import SentenceTransformer, util
# Ensure torch is imported if using tensors directly
import torch
# We need to import the T5 models
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- ADDED ---
# Load API key from .env file
load_dotenv()

# --- Initialize FastAPI App First ---
app = FastAPI()

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuring Hugging Face models...")

try:
    a = ollama.AsyncClient()
    print("Ollama client initialized.")
except Exception as ex: # Renamed to 'ex'
    print(f"Error initializing Ollama client: {ex}")
    a = None

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in .env file. GPT models will not work.")
        f = None
    else:
        f = AsyncOpenAI(api_key=openai_api_key)
        print("OpenAI client (f) initialized.")
except Exception as ex:
    print(f"Error initializing OpenAI client: {ex}")
    f = None

try:
    device = 'cpu'
    b = pipeline("question-answering", model="deepset/bart-base-squad2", device=device)
    c = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    d = pipeline("summarization", model="facebook/bart-large-cnn", device=device) 
    
    print("Loading T5 Question Generator (e)... this may take a moment.")
    e_model_name = "voidful/t5-base-qg-hl"
    e_tokenizer = T5Tokenizer.from_pretrained(e_model_name)
    e_model = T5ForConditionalGeneration.from_pretrained(e_model_name)
    e = {"tokenizer": e_tokenizer, "model": e_model, "device": device}
    e_model.to(device)
    print("T5 Question Generator (e) loaded.")

    print("Loading GPT-2 Generator (g)...")
    g = pipeline("text-generation", model="gpt2", device=device)
    print("GPT-2 Generator (g) loaded.")
    
    print(f"Hugging Face models loaded on device: {device}")

except Exception as ex:
    print(f"Error loading Hugging Face models: {ex}")
    b = None 
    c = None 
    d = None 
    e = None 
    g = None

BOOKS_BASE_DIR = "BOOKS"
if not os.path.exists(BOOKS_BASE_DIR):
    try:
        os.makedirs(BOOKS_BASE_DIR)
        print(f"Created base directory: {BOOKS_BASE_DIR}")
    except OSError as e:
        print(f"Error creating directory {BOOKS_BASE_DIR}: {e}")
        BOOKS_BASE_DIR = "." 

SUBJECT_ABBREVIATIONS = {
    "Geography": "geo",
    "History": "his",
    "Economics": "eco",
    "English": "eng",
    "Biology": "bio",
    "Political Science": "pol"
}

# --- Pydantic Models ---
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

# 1. Create a model for the *innermost* object
class QuestionRating(BaseModel):
    difficulty: str = ""
    remarks: str = ""

# 2. Update InvigilatorRatingData to use the new model
class InvigilatorRatingData(BaseModel):
    # This now correctly expects: {"mcqs": {"0": {"difficulty": "Easy", "remarks": "..."}}}
    ratings: Dict[str, Dict[str, QuestionRating]] = {}
    # This field name now correctly matches the React state
    overall_remarks: str = ""

# 3. Update the main model
class DifficultyRatings(BaseModel):
    class_num: str
    subject: str
    invigilator_data: Dict[str, InvigilatorRatingData] # e.g., {"invigilator1": {...}}


# --- Re-usable Helper Functions ---

def get_context_from_selection(class_num: str, subject: str) -> str:
    subject_abbr = SUBJECT_ABBREVIATIONS.get(subject)
    if not subject_abbr:
        print(f"Error: Subject abbreviation not found for {subject}")
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' is not configured.")

    class_dir = os.path.join(BOOKS_BASE_DIR, f"CLASS {class_num}")
    if not os.path.isdir(class_dir):
        print(f"Error: Class directory not found: {class_dir}")
        raise HTTPException(status_code=404, detail=f"Directory for Class {class_num} not found.")

    file_prefix = f"class_{class_num}_{subject_abbr}"
    found_file_path = None
    try:
        for filename in os.listdir(class_dir):
            if filename.lower().startswith(file_prefix.lower()) and filename.lower().endswith(".txt"):
                found_file_path = os.path.join(class_dir, filename)
                print(f"Found context file: {found_file_path}")
                break
    except FileNotFoundError:
        print(f"Error: Base directory not found: {BOOKS_BASE_DIR}")
        raise HTTPException(status_code=404, detail=f"Base directory '{BOOKS_BASE_DIR}' not found.")
    except Exception as e:
         print(f"Error listing directory {class_dir}: {e}")
         raise HTTPException(status_code=500, detail=f"Error accessing class directory: {e}")

    if not found_file_path:
        print(f"Error: Textbook file not found for prefix {file_prefix} in {class_dir}")
        raise HTTPException(status_code=404, detail=f"Textbook file for Class {class_num}, Subject {subject} not found (prefix: {file_prefix}).")

    try:
        with open(found_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                print(f"Warning: Context file is empty: {found_file_path}")
            return content
    except Exception as e:
        print(f"Error reading file {found_file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file '{os.path.basename(found_file_path)}': {e}")

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
    if not content_block: return q_list
    
    # Clean up leaked headers that might appear inside the block
    content_block = re.sub(r'(?:^|\n)\s*(?:\*\*)?Part \d.*?(?:\n|$)', '', content_block, flags=re.IGNORECASE)

    q_blocks = re.split(r'\n\s*(?=\d+\.)', content_block.strip())
    for block in q_blocks:
        if not block: continue 
        # Skip if it looks like a header
        if re.match(r'^\s*(?:\*\*)?Part', block, re.IGNORECASE): continue

        # Robustly split at "Answer:" if it exists on its own line
        parts = re.split(r'\n\s*Answer:.*', block, flags=re.IGNORECASE | re.DOTALL)
        question_text = parts[0] # Take only the part before "Answer:"

        clean_q = re.sub(r'^\d+\.\s*', '', question_text.strip()).strip()
        clean_q = re.sub(r'^\s*Here.*?:\s*\n?', '', clean_q, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Double check for headers inside the specific question string
        if "**Options:**" in clean_q: clean_q = clean_q.split("**Options:**")[0]
        if "Part " in clean_q and len(clean_q) < 20: continue 

        if clean_q: 
            q_list.append({"question": clean_q, "answer": None})
    return q_list

async def _run_llm_call(model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    t = ""
    if model_name.startswith("gpt-"):
        if not f: raise HTTPException(status_code=503, detail="OpenAI client (f) is not available.")
        print(f"Sending prompt to OpenAI model: {model_name}")
        response = await f.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful quiz-generation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        t = response.choices[0].message.content.strip()

    elif model_name == "gpt2":
        if not g: raise HTTPException(status_code=503, detail="GPT-2 client (g) is not available.")
        print(f"Sending prompt to local GPT-2 model: {model_name}")
        loop = asyncio.get_running_loop()
        response_list = await loop.run_in_executor(
            None, 
            g, 
            prompt, 
            max_new_tokens=max_tokens, 
            num_return_sequences=1
        )
        full_text = response_list[0]['generated_text']
        if full_text.startswith(prompt):
            t = full_text[len(prompt):].strip()
        else:
            t = full_text.strip() 

    else: # Assume it's an Ollama model
        if not a: raise HTTPException(status_code=503, detail="Ollama client (a) is not available.")
        print(f"Sending prompt to Ollama model: {model_name}")
        response_data = await a.generate(model=model_name, prompt=prompt, stream=False)
        t = response_data["response"].strip()
    
    return t

async def _generate_fallback_questions(
    ctx: str, 
    q_type: str, 
    count: int, 
    model_name: str,
    parse_func: Callable[[str], List[Dict[str, Union[str, None]]]]
) -> List[Dict[str, Union[str, None]]]:
    
    print(f"Fallback: Generating {count} {q_type} questions using {model_name}.")
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
            print(f"Fallback Error: Model {model_name} returned empty response.")
            return []
        return parse_func(t)
    except Exception as e:
        print(f"Error during fallback generation for {q_type}: {e}")
        return []

# --- API Endpoints ---

@app.post("/generate-questions")
async def generate_questions(
    class_num: str = Form(...),
    subject: str = Form(...),
    counts_json: str = Form(...),
    model_name: str = Form("llama3.1")
):
    ctx = ""
    try: 
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as e: 
        raise e
    except Exception as e: 
        print(f"Unexpected error getting context: {e}")
        raise HTTPException(status_code=500, detail="Failed to load context data.")
    
    if not ctx: 
        print("Error: Context is empty. Cannot generate questions.")
        raise HTTPException(status_code=404, detail="Context file is empty or unreadable. Cannot generate questions.")

    try: 
        counts = QuestionCounts.model_validate_json(counts_json)
    except ValidationError as e: 
        print(f"Error validating question counts: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON for question counts: {e}")

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

**CRITICAL REQUIREMENT:** You MUST generate questions for all three parts. Do not skip any part.
You must use the exact **Part X:** headers (e.g., `**Part 1: Multiple Choice Questions**`).
"""
    
    t = "" 
    d = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    try:
        t = await _run_llm_call(model_name, p, max_tokens=2048)
            
        print("="*50)
        print(f"RAW {model_name} RESPONSE (variable 't'):")
        print(t)
        print("="*50)

        # --- FIXED PARSING LOGIC ---
        # Use flexible splitting based on Part numbers regardless of titles
        def get_part_start_index(text, part_num):
            # Matches "Part 1", "**Part 1", "** Part 1", etc.
            pattern = r'(?:^|\n)\s*(?:\*\*)?\s*Part\s*' + str(part_num)
            match = re.search(pattern, text, re.IGNORECASE)
            return match.start() if match else -1

        idx_1 = get_part_start_index(t, 1)
        idx_2 = get_part_start_index(t, 2)
        idx_3 = get_part_start_index(t, 3)

        mcq_content = ""
        fib_content = ""
        sub_content = ""
        total_len = len(t)

        # 1. Extract MCQs
        if idx_1 != -1:
            end_of_1 = idx_2 if idx_2 != -1 else (idx_3 if idx_3 != -1 else total_len)
            mcq_content = t[idx_1:end_of_1]

        # 2. Extract Fill in Blanks
        if idx_2 != -1:
            end_of_2 = idx_3 if idx_3 != -1 else total_len
            fib_content = t[idx_2:end_of_2]

        # 3. Extract Subjective
        if idx_3 != -1:
            sub_content = t[idx_3:]

        d["mcqs"] = parse_questions_from_block(mcq_content)
        d["fill_in_the_blanks"] = parse_questions_from_block(fib_content)
        d["subjective"] = parse_questions_from_block(sub_content)

        if not d["mcqs"] and counts.mcqs > 0:
            print(f"WARNING: Main model failed to generate MCQs. Running fallback...")
            d["mcqs"] = await _generate_fallback_questions(
                ctx, "mcqs", counts.mcqs, model_name, parse_questions_from_block
            )
        if not d["fill_in_the_blanks"] and counts.fill_in_the_blanks > 0:
            print(f"WARNING: Main model failed to generate Fill in the Blanks. Running fallback...")
            d["fill_in_the_blanks"] = await _generate_fallback_questions(
                ctx, "fill_in_the_blanks", counts.fill_in_the_blanks, model_name, parse_questions_from_block
            )
        if not d["subjective"] and counts.subjective > 0:
            print(f"WARNING: Main model failed to generate Subjective. Running fallback...")
            d["subjective"] = await _generate_fallback_questions(
                ctx, "subjective", counts.subjective, model_name, parse_questions_from_block
            )
        
        if not d["mcqs"] and not d["fill_in_the_blanks"] and not d["subjective"]:
            print(f"WARNING: All generation failed, including fallbacks.")
        
        print(f"Final Parsed questions - MCQs: {len(d['mcqs'])}, FIB: {len(d['fill_in_the_blanks'])}, Subj: {len(d['subjective'])}")
        return {"questions": d, "context": ctx}
    
    except Exception as e: 
        print(f"Unexpected error during question generation: {e}")
        print(f"Raw response (if available): {t}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@app.post("/generate-bart-question")
async def generate_bart_question(
    class_num: str = Form(...),
    subject: str = Form(...),
    q_type: str = Form(...) 
):
    if not e: 
        raise HTTPException(status_code=503, detail="T5/BART Question Generator (e) not available.")
    print(f"--- Generating ONE question with T5/BART (model 'e') ---")
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        print(f"Unexpected error getting context for T5 QG: {ex}")
        raise HTTPException(status_code=500, detail="Failed to load context data.")
    if not ctx:
        raise HTTPException(status_code=400, detail="Context is required to regenerate question.")
    try:
        loop = asyncio.get_running_loop()
        new_q = await loop.run_in_executor(
            None,
            run_t5_qg, 
            ctx[:4000], 
            e 
        )
        if not new_q:
            raise Exception("Model 'e' returned an empty question.")
        print(f"T5/BART Generated Question: {new_q}")
        return {"new_question": new_q}
    except Exception as ex:
        print(f"Error during T5/BART question generation: {ex}")
        raise HTTPException(status_code=500, detail=f"T5/BART model failed: {str(ex)}")


@app.post("/generate-answers")
async def generate_answers(
    questions_json: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...),
    model_name: str = Form("llama3.1")
):
    if not a and not f and not g: raise HTTPException(status_code=503, detail="No generation clients available.")
    print(f"Generating answers for Class {class_num}, Subject {subject} using model {model_name}")
    try: ctx = get_context_from_selection(class_num, subject)
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected error getting context for answers: {e}"); raise HTTPException(status_code=500, detail="Failed to load context data.")
    if not ctx: print("Error: Cannot generate answers without context."); raise HTTPException(status_code=400, detail="Context file is empty or unreadable.")
    try: payload = QuestionsPayload.model_validate_json(questions_json); q_data = payload.questions
    except ValidationError as e: print(f"Error validating input questions JSON: {e}"); raise HTTPException(status_code=400, detail=f"Invalid JSON format for questions: {e}")
    
    answered_data = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}
    
    for q_type, q_list in q_data.items():
        if not q_list: print(f"Skipping empty question type: {q_type}"); continue
        prompts = []
        original_questions = []
        for i, item in enumerate(q_list):
            original_questions.append(item.question)
            clean_q = item.question
            if isinstance(clean_q, str):
                # Strip options and headers aggressively for the Prompt
                if "**Options:**" in clean_q: clean_q = clean_q.split("**Options:**")[0]
                elif "\nA)" in clean_q: clean_q = clean_q.split("\nA)")[0]
                
                # --- FIXED: Remove "Part 1:" headers that stuck to the question ---
                clean_q = re.sub(r'^(?:\*\*)?Part \d.*?(?:\*\*)?\n?', '', clean_q, flags=re.IGNORECASE)

                prefixes_patterns = [ r"^Here.*?text:", r"^Here is a new question:", r"^\*\*Question:\*\*", r"^\d+\.\s*" ]
                for pattern in prefixes_patterns: clean_q = re.sub(pattern, "", clean_q, flags=re.IGNORECASE).strip()
                clean_q = clean_q.replace("**", "").strip()
            else: clean_q = "Invalid question format"
            prompts.append(f"{i+1}. {clean_q}")
        
        joined_prompts = "\n".join(prompts)
        p_template = ""
        # --- FIXED: Explicit instructions to NOT repeat questions ---
        if q_type == "mcqs": p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: For each numbered question, provide ONLY the correct option letter and text (e.g., '1. C) Paris'). DO NOT repeat the question text. Format: 'Number. Option'"
        elif q_type == "fill_in_the_blanks": p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: For each numbered question, provide ONLY the single missing word or short phrase. DO NOT repeat the question. Format: '1. Answer'"
        else: p_template = f"Context:\n{ctx}\n\nQuestions:\n{joined_prompts}\n\nTask: Answer each numbered question in detail based on the context. Start with the number. DO NOT repeat the question text."
        
        raw_answers = ""
        try:
            raw_answers = await _run_llm_call(model_name, p_template, max_tokens=1024)
            print(f"Received answers for {q_type} from {model_name}.")
            
            # Cleanup markdown blocks
            raw_answers = re.sub(r'```[\w\s]*\n?', '', raw_answers); raw_answers = re.sub(r'\n?```', '', raw_answers)
            raw_answers = re.sub(r'^\*+\s*|\s*\*+$', '', raw_answers).strip()
            
            # Cleanup preamble like "Here are the answers:"
            raw_answers = re.sub(r'^(.*?)(?=1[\.\)])', '', raw_answers, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # --- FIXED: Robust split for 1. or 1) ---
            split_answers = re.split(r'\n\s*(?=\d+[\.\)])', raw_answers)
            
            parsed_answers = {}
            for ans_block in split_answers:
                # Match '1. Answer' or '1) Answer'
                match = re.match(r'(\d+)[\.\)]\s*(.*)', ans_block.strip(), re.DOTALL)
                if match:
                    q_num = int(match.group(1))
                    ans_text = match.group(2).strip()
                    ans_text = re.sub(r'^\*+\s*|\s*\*+$', '', ans_text).strip()
                    
                    # --- FIXED: Check if answer repeats the question (heuristic) ---
                    # If the answer starts with "What is..." or is unusually long compared to the question
                    if ans_text.lower().startswith("what") or ans_text.lower().startswith("why") or ans_text.lower().startswith("how"):
                        # Try to split by '?' if it exists, assuming answer comes after
                        if "?" in ans_text:
                            parts = ans_text.split("?")
                            if len(parts) > 1: ans_text = parts[-1].strip()
                    
                    parsed_answers[q_num - 1] = ans_text
            
            temp_list = []
            for i, original_q_text in enumerate(original_questions):
                answer = parsed_answers.get(i, "No answer generated or parsing failed.")
                temp_list.append(QuestionItem(question=original_q_text, answer=answer))
            answered_data[q_type] = temp_list
        except Exception as e: 
            print(f"Unexpected error generating answers for {q_type}: {e}"); 
            print(f"Raw response for {q_type} (if available): {raw_answers}")
            answered_data[q_type] = [QuestionItem(question=q, answer=f"Error generating answer: {str(e)}") for q in original_questions]
    
    print("Finished generating all answers."); return answered_data


@app.post("/evaluate-accuracy")
async def evaluate_accuracy(
    question: str = Form(...), 
    generated_answer: str = Form(...),
    q_type: str = Form(...),
    class_num: str = Form(...),
    subject: str = Form(...)
):
    if not b or not c: raise HTTPException(status_code=503, detail="Evaluation models not available.")
    print(f"Evaluating accuracy for: Q='{question[:50]}...', A='{generated_answer[:50]}...'")
    try: ctx = get_context_from_selection(class_num, subject)
    except HTTPException as e: print(f"Failed to get context for evaluation: {e.detail}"); return { "generated_answer": generated_answer, "bert_answer": f"Error loading context: {e.detail}", "similarity_score": "Error" }
    except Exception as e: print(f"Unexpected error getting context for evaluation: {e}"); return { "generated_answer": generated_answer, "bert_answer": "Unexpected error loading context.", "similarity_score": "Error" }
    if not ctx: print("Error: Context is empty, cannot evaluate accuracy."); return { "generated_answer": generated_answer, "bert_answer": "Error: Context unavailable", "similarity_score": 0.0 }
    
    if q_type == "subjective":
        try:
            if not generated_answer or len(generated_answer) < 5: score = 0.0
            else:
                emb1 = c.encode(ctx, convert_to_tensor=True); emb2 = c.encode(generated_answer, convert_to_tensor=True)
                if not isinstance(emb1, torch.Tensor) or not isinstance(emb2, torch.Tensor) or emb1.ndim == 0 or emb2.ndim == 0: score = 0.0
                else: sim = util.pytorch_cos_sim(emb1, emb2); score = sim.item()
            print(f"Subjective similarity score: {score:.4f}")
            return { "generated_answer": generated_answer, "bert_answer": "N/A (Relevance Score)", "similarity_score": score }
        except Exception as e: print(f"Error during subjective eval: {e}"); return { "generated_answer": generated_answer, "bert_answer": "N/A (Relevance Score)", "similarity_score": "Error" }
    else:
        loop = asyncio.get_running_loop(); bert_answer = "Error during BART/BERT processing."; score = 0.0
        try:
            if not question: raise ValueError("Question cannot be empty for QA evaluation.")
            bert_result = await loop.run_in_executor( None, b, {'question': question, 'context': ctx} )
            bert_answer = bert_result.get("answer", "BART could not find an answer.")
            print(f"BART answer found: '{bert_answer}'")
        except Exception as e: print(f"Error during BART QA exec: {e}")
        try:
            if not generated_answer or not bert_answer or bert_answer.startswith("Error") or bert_answer.startswith("BART could"): score = 0.0
            else:
                emb1 = c.encode(generated_answer, convert_to_tensor=True); emb2 = c.encode(bert_answer, convert_to_tensor=True)
                if not isinstance(emb1, torch.Tensor) or not isinstance(emb2, torch.Tensor) or emb1.ndim == 0 or emb2.ndim == 0: score = 0.0
                else: sim = util.pytorch_cos_sim(emb1, emb2); score = sim.item()
            print(f"BART vs Generated similarity score: {score:.4f}")
        except Exception as e: print(f"Error during similarity calc: {e}"); score = "Error"
        return { "generated_answer": generated_answer, "bert_answer": bert_answer, "similarity_score": score }

@app.post("/regenerate-answer")
async def regenerate_answer(
    class_num: str = Form(...),
    subject: str = Form(...),
    question: str = Form(...), 
    original_answer: str = Form(...),
    human_evaluation: str = Form(...),
    bert_answer: str = Form(...), 
    q_type: str = Form(...),
    model_name: str = Form("llama3.1")
):
    if not a and not f and not g: raise HTTPException(status_code=503, detail="No generation clients available.")
    print(f"Regenerating answer for Q='{question[:50]}...' using model {model_name}")
    if q_type in ["mcqs", "fill_in_the_blanks"]:
        corrected_answer = bert_answer if bert_answer and not bert_answer.startswith("BART could") and not bert_answer.startswith("Error") else original_answer
        print(f"Returning BART answer as correction for {q_type}: '{corrected_answer}'")
        return {"new_answer": corrected_answer}
    elif q_type == "subjective":
        try: ctx = get_context_from_selection(class_num, subject)
        except HTTPException as e: raise e
        except Exception as e: print(f"Unexpected error getting context for regen answer: {e}"); raise HTTPException(status_code=500, detail="Failed to load context data.")
        if not ctx: raise HTTPException(status_code=400, detail="Context is required to regenerate subjective answer.")
        clean_q_for_prompt = question
        if isinstance(clean_q_for_prompt, str):
            if "**Options:**" in clean_q_for_prompt: clean_q_for_prompt = clean_q_for_prompt.split("**Options:**")[0]
            elif "\nA)" in clean_q_for_prompt: clean_q_for_prompt = clean_q_for_prompt.split("\nA)")[0]
            prefixes_patterns = [r"^Here.*?text:", r"^Here is a new question:", r"^\*\*Question:\*\*", r"^\d+\.\s*"]
            for pattern in prefixes_patterns: clean_q_for_prompt = re.sub(pattern, "", clean_q_for_prompt, flags=re.IGNORECASE).strip()
            clean_q_for_prompt = clean_q_for_prompt.replace("**", "").strip()
        else: clean_q_for_prompt = "Invalid question format provided"
        p = f"""
You are an expert teaching assistant tasked with correcting a detailed, subjective answer based *only* on the provided context.
**Full Context:**
---
{ctx}
---
**Question:**
{clean_q_for_prompt}
**Previous Flawed Answer:**
"{original_answer}"
**Human Feedback:**
The previous answer was marked as: **{human_evaluation}**.
**Your Task:**
Generate a new, improved, and comprehensive answer to the subjective question using only information from the Full Context.
**Output:**
Provide ONLY the new, corrected answer. Do not include introductory phrases.
"""
        new_answer = ""
        try:
            new_answer = await _run_llm_call(model_name, p, max_tokens=512)
            new_answer = re.sub(r'```[\w\s]*\n?', '', new_answer); new_answer = re.sub(r'\n?```', '', new_answer)
            new_answer = re.sub(r'^\*+\s*|\s*\*+$', '', new_answer).strip()
            intro_phrases = ["here is the new answer:", "here is the corrected answer:", "the corrected answer is:"]
            for phrase in intro_phrases:
                if new_answer.lower().startswith(phrase): new_answer = new_answer[len(phrase):].strip()
            if not new_answer: print("Error: Model returned empty string for regenerated answer."); raise HTTPException(status_code=500, detail="Model failed to regenerate a valid answer.")
            print("Received regenerated subjective answer.")
            return {"new_answer": new_answer}
        except Exception as e: 
            print(f"Unexpected error during answer regeneration: {e}")
            raise HTTPException(status_code=500, detail=f"Answer regeneration failed: {str(e)}")
    else: print(f"Error: Unknown question type for regeneration: {q_type}"); raise HTTPException(status_code=400, detail=f"Unknown question type for regeneration: {q_type}")

@app.post("/answer-questions-pdf")
async def answer_questions_pdf(req: PdfRequest):
    data = req.answered_questions
    if not data or not any(data.values()): print("Error: No answered questions provided for PDF generation."); raise HTTPException(status_code=400, detail="No answered questions provided.")
    buffer = BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=letter); styles = getSampleStyleSheet(); story = []
    try:
        story.append(Paragraph("Generated Answers", styles['h1'])); story.append(Spacer(1, 24))
        for q_type, q_list in data.items():
            if q_list:
                header = q_type.replace('_', ' ').title(); story.append(Paragraph(header, styles['h2']))
                for i, item in enumerate(q_list, 1):
                    q_text_raw = str(item.question) if item.question else "No Question Text Provided"
                    a_text_raw = str(item.answer) if item.answer else "No Answer Provided"
                    q_text_safe = q_text_raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    a_text_safe = a_text_raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    q_text_pdf = f"{i}. <b>{q_text_safe}</b>".replace('\n', '<br/>'); story.append(Paragraph(q_text_pdf, styles['Normal']))
                    story.append(Spacer(1, 6)); a_text_pdf = a_text_safe.replace('\n', '<br/>'); story.append(Paragraph(a_text_pdf, styles['Normal']))
                    story.append(Spacer(1, 12))
        print("Building PDF document..."); doc.build(story); buffer.seek(0); print("PDF built successfully.")
    except Exception as e: print(f"Error building PDF: {e}"); raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")
    return StreamingResponse(buffer, media_type="application/pdf", headers={ "Content-Disposition": "attachment; filename=generated_answers.pdf" })

@app.post("/evaluate-question-bert")
async def evaluate_question_bert(req: QuestionEvaluationRequest):
    if not b: raise HTTPException(status_code=503, detail="QA model (BART) not available.")
    print(f"Evaluating question (BART): '{req.question[:50]}...'")
    try:
        ctx = get_context_from_selection(req.class_num, req.subject)
        if not ctx: print("Error: Context unavailable for BART evaluation."); return { "question": req.question, "bert_confidence_score": 0.0, "inferred_answer": "Error: Context unavailable." }
        loop = asyncio.get_running_loop()
        bert_result = await loop.run_in_executor( None, b, {'question': req.question, 'context': ctx} )
        confidence_score = bert_result.get("score", 0.0)
        found_answer = bert_result.get("answer", "BART could not find an answer.")
        print(f"BART evaluation score: {confidence_score:.4f}, Answer: '{found_answer}'")
        return { "question": req.question, "bert_confidence_score": confidence_score, "inferred_answer": found_answer }
    except Exception as e: print(f"Error during BART evaluation execution: {e}"); return { "question": req.question, "bert_confidence_score": "Error", "inferred_answer": f"BART evaluation failed: {str(e)}" }

@app.post("/log-human-evaluation")
async def log_human_evaluation(log: HumanEvaluationLog):
    print(f"Human Feedback Received:")
    print(f"  Question Type: {log.q_type}")
    print(f"  Evaluation: {log.evaluation}")
    print(f"  Question: {log.question}")
    print("-" * 20)
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
    if not d: raise HTTPException(status_code=503, detail="BART summarizer model (d) not available.")
    print(f"--- Starting HYBRID Question Regeneration using {model_name} ---")
    
    try:
        ctx = get_context_from_selection(class_num, subject)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error getting context for regen question: {e}")
        raise HTTPException(status_code=500, detail="Failed to load context data.")
    
    if not ctx:
        raise HTTPException(status_code=400, detail="Context is required to regenerate question.")

    key_content = ""
    try:
        print("Asking BART (model 'd') to summarize context...")
        loop = asyncio.get_running_loop()
        context_chunk = ctx[:4000] 
        summary_result = await loop.run_in_executor(None, d, context_chunk, max_length=150, min_length=30, do_sample=False)
        if not summary_result:
            raise Exception("BART summarizer returned no result.")
        summary_text = summary_result[0]['summary_text']
        key_content = summary_text.split('.')[0] + '.'
        print(f"BART extracted key content: {key_content}")
    except Exception as e:
        print(f"BART summarization failed: {e}")
        sentences = ctx.split('.')
        fallback_index = min(5, len(sentences) - 1)
        key_content = sentences[fallback_index].strip() + '.'
        print(f"BART failed, using fallback content: {key_content}")

    if q_type == "mcqs":
        prompt_task = f"""
        You are a question generator. Turn the following fact into one (1) multiple-choice question (MCQ) with four options (A, B, C, D).
        Do NOT include the answer. Fact: "{key_content}"
        Output ONLY the question and options, like this:
        1. Question text?
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        """
    elif q_type == "fill_in_the_blanks":
        prompt_task = f"""
        You are a question generator. Turn the following fact into one (1) fill-in-the-blanks question.
        Replace a key word or phrase with '____'. Fact: "{key_content}"
        Output ONLY the question, like this:
        1. The fact's main point is ____.
        """
    else: # Subjective
        prompt_task = f"""
        You are a question generator. Turn the following fact into one (1) subjective (long-answer) question.
        Fact: "{key_content}"
        Output ONLY the question, like this:
        1. Explain the significance of...
        """

    new_q = ""
    try:
        new_q = await _run_llm_call(model_name, prompt_task, max_tokens=256)
        new_q = re.sub(r'```[\w\s]*\n?', '', new_q)
        new_q = re.sub(r'\n?```', '', new_q)
        new_q = re.sub(r'^\*+\s*|\s*\*+$', '', new_q).strip()
        prefixes_to_remove = ["here is the question:", "here is a new question:", "**Question:**", "Here is the multiple-choice question:"]
        for prefix in prefixes_to_remove:
            if new_q.lower().startswith(prefix.lower()): new_q = new_q[len(prefix):].strip()
        if not new_q:
            raise Exception("Model returned empty string.")
        print(f"{model_name} generated question: {new_q}")
        return {"new_question": new_q}
    except Exception as e:
        print(f"{model_name} question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"{model_name} failed to generate question: {e}")


@app.get("/")
async def root():
    status = { 
        "message": "Quiz Generator API is running", 
        "ollama_client_status": "Initialized" if a else "Failed",
        "openai_client_status": "Initialized" if f else "Failed (Check API Key)",
        "bart_qa_status": "Loaded" if b else "Failed", 
        "sentence_transformer_status": "Loaded" if c else "Failed",
        "bart_summarizer_status": "Loaded" if d else "Failed",
        "t5_qg_status": "Loaded" if e else "Failed",
        "gpt2_gen_status": "Loaded" if g else "Failed"
    }
    return status

@app.get("/check-models")
async def check_models():
    ollama_available = False; models_list = []
    if a:
        try: list_response = await a.list(); ollama_available = True; models_list = [m['name'] for m in list_response.get('models', [])]
        except Exception as e: print(f"Could not reach Ollama service: {e}"); ollama_available = False
    return { 
        "ollama_client": "Initialized" if a else "Failed", 
        "ollama_service_reachable": ollama_available, 
        "ollama_models_available": models_list, 
        "openai_client": "Initialized" if f else "Failed (Check API Key)",
        "qa_model_status": "Loaded (BART)" if b else "Failed", 
        "sentence_transformer": "Loaded" if c else "Failed",
        "bart_summarizer": "Loaded (BART)" if d else "Failed",
        "t5_qg_model": "Loaded" if e else "Failed",
        "gpt2_gen_model": "Loaded" if g else "Failed"
    }

@app.post("/submit-difficulty-ratings")
async def submit_difficulty_ratings(ratings_data: DifficultyRatings):
    print("--- Received Difficulty Ratings ---")
    print(f"Class: {ratings_data.class_num}")
    print(f"Subject: {ratings_data.subject}")
    
    # This line is now correct and uses the pydantic .model_dump_json() method
    print(f"Ratings Data: {ratings_data.model_dump_json(indent=2)}")
    
    # In a real application, you would save this data to a database
    
    return {"status": "success", "message": "Ratings logged."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
