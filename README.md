The Problem Statement

Every day in India, millions of patients walk out of hospitals clutching a medical report they cannot read. The report says "Type 2 Diabetes Mellitus with HbA1c 11.2%" — but the patient, a 58-year-old woman named Lakshmi who studied up to Class 8 and speaks only Tamil, has no idea what that means.

She doesn't know her blood sugar is dangerously high. She doesn't know she needs medicine every day. She doesn't know that ignoring this could cost her eyesight, her kidneys, or her life.

**This is not a rare edge case.** This is the reality for the majority of India's population:

-**22 official languages**, but medical reports are written in English
-**65% of the population** has education below Class 10
-**Patients are scared**, confused, and often too embarrassed to ask even if they ask the doctors are explaining it in a complex way
-**Misinformation fills the gap** — Due to the high useage of the social media the misinformation about an disease are more.

Our Solution MedBridge is a medical Explainable AI 

**MedBridge trains an AI to be the missing translator** — a compassionate doctor who takes a complex medical report and explains it to the patient in their own language, at their reading level, matching their emotional state.

---

#The Environment: Teaching AI to Be a Better Doctor

MedBridge is a **reinforcement learning environment** built on [Meta's OpenEnv](https://github.com/meta-pytorch/openenv) framework. It simulates the moment when a doctor sits down with a patient to explain their diagnosis.

### How an Episode Works

Think of each episode as one doctor-patient conversation:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  🔄 RESET — A new patient walks in                              │
│     → Lakshmi, 58, Tamil, Class 8 education, very scared        │
│     → Report: "HbA1c 11.2%, fasting glucose 340 mg/dL..."      │
│     → Diagnosis: Type 2 Diabetes (Serious)                      │
│                                                                 │
│  📝 STEP 1 — The AI explains the diagnosis                      │
│     → "அம்மா, உங்கள் இரத்தத்தில் சர்க்கரை அளவு                │
│        மிகவும் அதிகமாக இருக்கிறது..."                           │
│     → Environment scores: accuracy, simplicity, tone, language  │
│     → Patient asks: "நான் இன்னும் அரிசி சாப்பிடலாமா?"          │
│        ("Can I still eat rice?")                                │
│                                                                 │
│  💬 STEP 2 — The AI answers the follow-up                       │
│     → "சிறிய அளவில் சாதம் சாப்பிடலாம்..."                      │
│     → Environment scores: follow-up quality                     │
│     → Final reward: weighted combination of all 5 scores        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Patient Diversity

Every episode generates a **unique patient** from a pool of realistic Indian demographics:

| Dimension | Options |
|-----------|---------|
| **Languages** | Tamil, Hindi, Telugu, Kannada, Marathi |
| **Education** | Class 5, Class 8, Class 10, Graduate |
| **Emotional State** | Calm (1-3) → Anxious (4-6) → Scared (7-8) → Panicked (9-10) |
| **Location** | Rural, Semi-urban, Urban |
| **Age** | 30 – 75 years |

### Medical Reports

The environment includes **20 carefully researched medical conditions** across 5 categories:

| Category | Conditions | Severity |
|----------|-----------|----------|
| **Chronic** | Diabetes, Hypertension, Hypothyroidism, CKD, Asthma | Moderate–Serious |
| **Acute** | Appendicitis, Dengue, Pneumonia, Heart Attack, Stroke | Critical |
| **Cancer** | Breast, Cervical, Oral, Colorectal, Thyroid | Moderate–Serious |
| **Mental Health** | Depression, Anxiety, Bipolar Disorder | Moderate |
| **Pediatric** | Childhood Asthma, Type 1 Diabetes | Moderate–Serious |

Each report contains real clinical language (e.g., *"Acute ST Elevation Myocardial Infarction, anterior wall. Troponin I 18.4 ng/mL"*), key facts the AI must communicate, and **forbidden claims** the AI must never make (e.g., *"you are completely cured"*).

---

## ⚖️ The Reward System: How We Know the AI Is Actually Helping

Most chatbots optimize for one thing — sounding fluent. But a medical communicator needs to be **accurate**, **simple**, **emotionally appropriate**, **in the right language**, and **responsive to questions**. That's why MedBridge uses **5 independent reward dimensions**:

### 1. 🎯 Medical Accuracy (Weight: 30%)
> *Did the AI mention all the important medical facts without saying anything dangerous?*

The environment checks whether the AI's explanation contains the key facts from the report (e.g., "blood sugar is very high", "need to take medicine every day") and ensures it never makes a forbidden claim (e.g., "you are completely cured"). If any forbidden claim appears, the accuracy score drops to **zero instantly** — no exceptions.

### 2. 📖 Simplicity (Weight: 20%)
> *Would the patient actually understand this?*

We use the [Flesch-Kincaid readability formula](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) (via `textstat`) to measure the reading grade level of the AI's explanation. A Class 8 patient needs language at grade 6 or below. If the AI uses words like "peripheral neuropathy" instead of "numbness in hands and feet," this score drops.

### 3. 💛 Emotional Tone (Weight: 20%)
> *Is the AI being warm enough for a scared patient?*

A terrified patient with an emotional state of 8/10 needs warmth and reassurance. A calm patient needs clear, direct information. We use a **RoBERTa-based sentiment classifier** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to analyze the AI's tone and match it to the patient's emotional needs.

### 4. 🌐 Language Match (Weight: 20%)
> *Did the AI actually respond in Tamil, or did it default to English?*

We use `langdetect` to verify that the AI's response is in the patient's requested language. A Tamil patient getting an English response scores **zero** on this dimension, no matter how good the explanation is.

### 5. ❓ Follow-up Quality (Weight: 10%)
> *When the patient asked "Can I still eat rice?", did the AI give a real answer?*

Each diagnosis has a specific follow-up question with expected answer keywords. The environment checks whether the AI's response actually addresses the question.

### Anti-Hacking Safeguards

We built in protections against reward gaming:

- **Forbidden claims** → Instant zero accuracy (and total reward = 0)
- **Independent rewards** → Can't game one dimension to compensate for another
- **Real NLP models** → Tone and language are checked by ML classifiers, not regex
- **Objective readability** → Simplicity is measured by standardized formulas

---

## 🧠 The Model: Teaching Qwen to Be a Compassionate Doctor

### Base Model

We use [**Qwen 2.5-3B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — a 3-billion parameter instruction-following model by Alibaba. We chose it because:

- **Multilingual** — Strong performance in Hindi, and reasonable in other Indian languages
- **Efficient** — 3B parameters fits on a free-tier Colab T4 GPU with 4-bit quantization
- **Instruction-tuned** — Already knows how to follow complex prompts

### Fine-tuning: LoRA + GRPO

We don't retrain the entire 3B model — that would require enterprise-grade hardware. Instead, we use **LoRA (Low-Rank Adaptation)** to train only a small set of adapter weights:

| Parameter | Value |
|-----------|-------|
| **Method** | QLoRA (4-bit quantized LoRA) |
| **Rank (r)** | 16 |
| **Alpha** | 16 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainable Parameters** | ~114 MB (vs 6 GB full model) |
| **Dropout** | 0 |

The result? A `adapter_model.safetensors` file of just **114 MB** that transforms a general-purpose chatbot into a compassionate medical communicator.

### Training Algorithm: GRPO

We train using **Group Relative Policy Optimization (GRPO)** — a reinforcement learning algorithm from the [TRL library](https://github.com/huggingface/trl) that's particularly well-suited for language models.

Here's how GRPO works in our setup:

1. **Sample** — For each patient scenario, the model generates 4 different explanations
2. **Score** — Each explanation is scored by our 5 reward functions
3. **Compare** — Within each group, better explanations get higher advantage scores
4. **Update** — The model's weights are nudged toward the better explanations

This is different from supervised fine-tuning where you need "gold standard" answers. With GRPO, the model **discovers its own best strategies** through trial and error — learning which words, which tone, and which level of detail works best for each type of patient.

### Training Configuration

| Setting | Value |
|---------|-------|
| **Hardware** | Google Colab T4 GPU (16 GB VRAM) |
| **Epochs** | 3 |
| **Batch Size** | 2 (per device) |
| **Gradient Accumulation** | 4 steps |
| **Learning Rate** | 5e-6 |
| **Max Completion Length** | 512 tokens |
| **Generations per Prompt** | 4 |
| **Temperature** | 0.7 |
| **Training Scenarios** | 200 diverse patient cases |
| **Optimizer** | Unsloth-optimized AdamW |
| **Precision** | FP16 |
| **Tracking** | Weights & Biases |

---

### Train Your Own Model

Open the training notebook in Google Colab:

→ [**Training Notebook**](https://colab.research.google.com/drive/1r2JGgFEcvQidzyksnTtBwSNUV_KFG0_0?usp=sharing)

---

## 🏗️ Project Structure

```
MedBridge/
├── medbridge/                          # Core environment package
│   ├── server/
│   │   ├── medbridge_environment.py    # OpenEnv Environment (2-step episode)
│   │   └── app.py                      # FastAPI server
│   ├── models.py                       # Action / Observation schemas
│   ├── patients.py                     # Random patient profile generator (20 names × 5 languages)
│   ├── reports.py                      # 20 medical report templates with key facts
│   ├── followups.py                    # Follow-up Q&A rules per diagnosis
│   ├── rewards.py                      # 5 independent reward functions
│   ├── inference.py                    # Qwen2.5 + LoRA inference engine
│   ├── Dockerfile                      # Container build for HF Spaces
│   └── README.md                       # HF Space metadata
│
├── Model/                              # Trained LoRA adapter
│   ├── adapter_config.json             # LoRA configuration (r=16, 7 target modules)
│   ├── adapter_model.safetensors       # Trained weights (114 MB)
│   └── train.ipynb                     # Executed training notebook with outputs
│
├── training/
│   ├── train.ipynb                     # Clean training notebook (for Colab)
│   └── train.py                        # Training script (standalone)
│
├── ui/                                 # React frontend
│   └── src/
│       ├── App.jsx                     # Main chat interface
│       └── components/
│           └── ChatInterface.jsx       # Patient ↔ Doctor chat UI
│
└── pyproject.toml                      # Python dependencies
```

---

## 🤝 Why This Matters

This isn't just an academic exercise. The problem MedBridge addresses is **happening right now**, in every government hospital, every PHC, every rural clinic across India.

- **A grandmother in rural Tamil Nadu** gets a diabetes diagnosis she can't read
- **A factory worker in Pune** doesn't understand his hypertension report is urgent
- **A scared mother in Hyderabad** doesn't know her child's asthma can be managed

MedBridge is our attempt to build the bridge between what doctors say and what patients understand — using reinforcement learning to train AI that doesn't just translate words, but communicates with empathy, accuracy, and cultural sensitivity.

---

## Reference Links

1. GitHub Repository: https://github.com/Aries0x/Medbridge
2. Google Colab Notebook: https://colab.research.google.com/drive/1r2JGgFEcvQidzyksnTtBwSNUV_KFG0_0?usp=sharing
3. Hugging Face Space: https://huggingface.co/spaces/Nermal007/medbridge/tree/main
4. Blog linK: https://huggingface.co/spaces/Nermal007/medbridge/blob/main/Blog.md


## 🛠️ How to Run the Project (Step by Step)

### Prerequisites

Make sure you have these installed on your machine:

- **Python 3.10+** → [Download](https://www.python.org/downloads/)
- **Node.js 18+** → [Download](https://nodejs.org/)
- **Git** → [Download](https://git-scm.com/)
- **A HuggingFace account** (free) → [Sign up](https://huggingface.co/join)

### Option 1: Run Everything Locally

#### Step 1 — Clone the repository

```bash
git clone https://github.com/Aries0x/Medbridge.git
cd Medbridge
```

#### Step 2 — Install Python dependencies

```bash
pip install -e ./medbridge
pip install torch transformers peft accelerate textstat langdetect
pip install fastapi uvicorn requests
```

#### Step 3 — Download the base model (one-time, ~6 GB)

The LoRA adapter (114 MB) is already in the `Model/` folder, but it needs the base Qwen model:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='base_model')"
```

> ⏳ This downloads ~6 GB. It only needs to happen once — the files are cached after that.

#### Step 4 — Start the backend server

```bash
python -m medbridge.server.app --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 5 — Start the frontend (in a new terminal)

```bash
cd ui
npm install
npm run dev
```

You should see:
```
VITE ready in 400ms
➜  Local: http://localhost:5173/
```

#### Step 6 — Open in browser

Go to **http://localhost:5173/** — you'll see the MedBridge chat interface. You play as the **patient** and the AI model acts as your **doctor**. Ask questions about your diagnosis and the model will respond.

---

### Option 2: Use the Live HuggingFace Space (No Installation)

If you don't want to install anything, you can connect directly to our deployed environment:

```python
pip install medbridge
```

```python
from medbridge import MedbridgeAction, MedbridgeEnv

with MedbridgeEnv(base_url="https://Nermal007-medbridge.hf.space").sync() as env:
    result = env.reset()
    obs = result.observation
    print(f"Patient: {obs.patient_profile['name']}")
    print(f"Diagnosis: {obs.diagnosis_name}")

    # Send an explanation
    result = env.step(MedbridgeAction(
        explanation="Your blood sugar is very high. You need medicine daily."
    ))
    print(f"Follow-up: {result.observation.followup_question}")
    print(f"Scores: {result.observation.reward_breakdown}")
```

---

### Option 3: Train Your Own Model (Google Colab)

1. Open the [Training Notebook](https://colab.research.google.com/drive/1r2JGgFEcvQidzyksnTtBwSNUV_KFG0_0?usp=sharing) in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells — training takes ~30–45 minutes on a T4
4. The trained adapter will be saved to `./medbridge_grpo_model/`
5. Download `adapter_model.safetensors` and place it in the `Model/` folder

---

## 📝 License

Built on [OpenEnv](https://github.com/meta-pytorch/openenv) (BSD License) for the **OpenEnv Hackathon India 2026**.
