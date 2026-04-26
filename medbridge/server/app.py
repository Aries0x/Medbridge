"""
MedBridge API Server
====================
The model acts as a DOCTOR. The user acts as a PATIENT.

Flow:
  1. POST /reset  → Generate patient case, model gives initial explanation
  2. POST /chat   → User asks question (as patient), model answers (as doctor)
  3. GET /model_status → Check if model is loaded
"""

import sys
import os
import asyncio
import traceback
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patients import generate_patient  # type: ignore
from reports import get_random_report  # type: ignore

# --- FastAPI App ---
app = FastAPI(title="MedBridge AI — Doctor Chat", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session state ---
_current_patient = None
_current_report = None
_chat_history: List[Dict[str, str]] = []

SYSTEM_PROMPT = (
    "You are a compassionate medical communicator in India. "
    "Explain medical diagnoses simply and empathetically in the patient's language. "
    "Answer all patient questions clearly and honestly."
)


# --- Model Inference ---
inference_engine = None
try:
    from inference import MedbridgeInference  # type: ignore
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model"))
    inference_engine = MedbridgeInference(model_path=model_dir)
except Exception as e:
    print(f"Warning: Could not import inference module: {e}")

_model_executor = ThreadPoolExecutor(max_workers=1)
_model_loading = False
_model_ready = False


# --- Request Models ---
class ChatRequest(BaseModel):
    message: str


# --- Endpoints ---

@app.post("/reset")
async def reset_env():
    """Generate a new patient case and return it."""
    global _current_patient, _current_report, _chat_history

    _current_patient = generate_patient()
    _current_report = get_random_report()
    _chat_history = []

    return {
        "patient": _current_patient,
        "report": {
            "medical_report": _current_report.get("report_text", ""),
            "diagnosis_name": _current_report.get("diagnosis_name", ""),
            "severity": _current_report.get("severity", ""),
        },
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    User sends a message as the PATIENT.
    Model responds as the DOCTOR.
    """
    global _model_loading, _model_ready, _chat_history

    if inference_engine is None:
        return {"error": "Model not available. Check server logs."}

    if _current_patient is None:
        return {"error": "No active session. Call /reset first."}

    # Build the context message with patient info (for the first message)
    if len(_chat_history) == 0:
        context = (
            f"Patient: {_current_patient.get('name')}, "
            f"Age: {_current_patient.get('age')}, "
            f"Language: {_current_patient.get('language')}, "
            f"Education: {_current_patient.get('education_level')}\n\n"
            f"Medical Report:\n{_current_report.get('report_text', '')}\n\n"
            f"Diagnosis: {_current_report.get('diagnosis_name', '')}\n\n"
            f"The patient is here. Greet them and explain their diagnosis "
            f"in {_current_patient.get('language')}. Keep it simple."
        )
        _chat_history.append({"role": "user", "content": context})

    # Add the patient's new message
    _chat_history.append({"role": "user", "content": request.message})

    # Build full message list with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _chat_history

    def _do_chat():
        global _model_loading, _model_ready
        _model_loading = True
        try:
            result = inference_engine.chat(messages)
            _model_ready = True
            _model_loading = False
            return result
        except Exception as e:
            _model_loading = False
            raise e

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(_model_executor, _do_chat)
        # Add the doctor's response to history
        _chat_history.append({"role": "assistant", "content": response})
        return {"response": response}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/model_status")
async def model_status():
    if inference_engine is None:
        return {"status": "unavailable"}
    if _model_ready:
        return {"status": "ready"}
    if _model_loading:
        return {"status": "loading"}
    return {"status": "idle"}


# --- Entry Point ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
