---
title: MedBridge Environment Server
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🏥 MedBridge — Bridging the Gap Between What Doctors Say and What Patients Understand

**A 2-step medical communication RL environment that trains LLMs to explain complex diagnoses to patients in their native Indian language, at their reading level, matching their emotional state.**

> **Theme**: World Modeling → Professional Tasks  
> **OpenEnv Hackathon India 2026**

---

## 🔍 The Problem

In India, **800 million people** receive medical diagnoses written in complex English terminology. Most patients:

- Cannot read or understand medical jargon ("bilateral infiltrates", "HbA1c > 7%")
- Speak regional languages (Tamil, Hindi, Telugu, Kannada, Marathi)
- Have low literacy levels (Class 5–10 education)
- Are emotionally overwhelmed and scared
- Have no one to explain what the report actually means

**Result**: Patients ignore critical diagnoses, self-medicate dangerously, miss follow-ups, and trust WhatsApp forwards over doctors.

**MedBridge trains an LLM to be the missing translator** — converting medical jargon into empathetic, culturally appropriate explanations.

---

## 🎮 How the Environment Works

MedBridge is a **2-step episodic environment** with 5 independent reward dimensions:

```
┌──────────────────────────────────────────────────────────┐
│                    EPISODE FLOW                          │
│                                                          │
│  reset() → Patient Profile + Medical Report              │
│     │                                                    │
│     ▼                                                    │
│  step(1) → Agent sends EXPLANATION                       │
│     │       Environment scores: Accuracy, Simplicity,    │
│     │                          Tone, Language             │
│     │       Returns: Follow-up question from patient     │
│     ▼                                                    │
│  step(2) → Agent sends FOLLOW-UP ANSWER                  │
│             Environment scores: Follow-up quality        │
│             Returns: Full reward breakdown + done=True    │
└──────────────────────────────────────────────────────────┘
```

### Observation Space (on reset)

| Field | Example |
|-------|---------|
| `patient_profile.name` | "Lakshmi" |
| `patient_profile.age` | 58 |
| `patient_profile.language` | "Tamil" |
| `patient_profile.education_level` | "Class 8" |
| `patient_profile.emotional_state` | 8 (scared) |
| `medical_report` | "HbA1c: 8.2%, fasting glucose 180 mg/dL..." |
| `diagnosis_name` | "Type 2 Diabetes" |
| `severity` | "Moderate" |

### Action Space

| Step | Field | Description |
|------|-------|-------------|
| 1 | `explanation` | Plain-language medical explanation for the patient |
| 2 | `followup_answer` | Answer to the patient's follow-up question |

### Reward Function (5 Independent Dimensions)

| Reward | Weight | How It's Computed |
|--------|--------|-------------------|
| **Accuracy** | 30% | Key medical facts present + no forbidden claims |
| **Simplicity** | 20% | Reading grade ≤ patient's education level (textstat) |
| **Tone** | 20% | Emotional warmth matches patient's fear (RoBERTa sentiment classifier) |
| **Language** | 20% | Response in correct regional language (langdetect) |
| **Follow-up** | 10% | Follow-up answer covers required concepts |

**Anti-hacking**: If accuracy = 0 (key facts missing), total reward = 0.0 regardless of other scores. Forbidden medical claims (e.g., "this cures cancer") cause instant accuracy = 0.

---

## 🚀 Quick Start

### Connect to Deployed Environment

```python
from medbridge import MedbridgeAction, MedbridgeEnv

# Connect to HuggingFace Space
with MedbridgeEnv(base_url="https://Nermal007-medbridge.hf.space").sync() as env:
    # Reset — get patient + diagnosis
    result = env.reset()
    obs = result.observation
    print(f"Patient: {obs.patient_profile['name']}, {obs.patient_profile['language']}")
    print(f"Diagnosis: {obs.diagnosis_name} ({obs.severity})")

    # Step 1 — Send explanation
    result = env.step(MedbridgeAction(
        explanation="Your blood sugar is high. You need to take medicine daily..."
    ))
    print(f"Follow-up Q: {result.observation.followup_question}")
    print(f"Partial scores: {result.observation.reward_breakdown}")

    # Step 2 — Answer follow-up
    result = env.step(MedbridgeAction(
        followup_answer="Yes, you can eat rice but in small portions..."
    ))
    print(f"Final reward: {result.reward}")
    print(f"Full breakdown: {result.observation.reward_breakdown}")
```

### Run Locally

```bash
cd medbridge/
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 📊 Environment Design

### Patient Diversity (20 profiles × 5 languages × 4 education levels)

- **Languages**: Tamil, Hindi, Telugu, Kannada, Marathi
- **Education**: Class 5 → Class 10 → Graduate
- **Emotional states**: Calm (1-3) → Anxious (4-6) → Scared (7-8) → Panicked (9-10)
- **Locations**: Rural, Semi-urban, Urban

### Medical Report Templates (20 diagnoses)

| ID | Diagnosis | Severity | Example Follow-up |
|----|-----------|----------|-------------------|
| 1 | Type 2 Diabetes | Moderate | "Can I still eat rice?" |
| 2 | Hypertension | Moderate | "Can I stop medicine if BP is normal?" |
| 3 | Pulmonary Tuberculosis | Serious | "Will my family get TB?" |
| 4 | Chronic Kidney Disease | Serious | "Do I need dialysis?" |
| 5 | Breast Cancer Stage 2A | Serious | "Will I lose my hair?" |
| ... | ... | ... | ... |
| 20 | Bilateral Pneumonia | Critical | "Can I go home tomorrow?" |

Each report includes: technical diagnosis text, key facts, forbidden claims, and lifestyle advice.

### Curriculum Learning (3 Phases)

| Phase | Severity | Reports | Purpose |
|-------|----------|---------|---------|
| Easy | Mild | Anemia, UTI, GERD | Build basic communication skills |
| Medium | Moderate | Diabetes, Hypertension, Thyroid | Handle common conditions |
| Hard | Serious/Critical | Cancer, CKD, Heart Attack | Master difficult conversations |

---

## 🔒 Anti-Reward-Hacking Measures

1. **Forbidden Claims Check**: Certain medical claims (e.g., "this cures cancer", "stop all medication") instantly zero out accuracy
2. **Independent Rewards**: 5 separate scoring dimensions cannot be gamed by optimizing one
3. **Sentiment Analysis via RoBERTa**: Tone is scored by a dedicated classifier (cardiffnlp/twitter-roberta-base-sentiment-latest), not by regex
4. **Reading Level via textstat**: Simplicity is measured objectively by Flesch-Kincaid grade level
5. **Language Detection**: langdetect verifies the response is in the correct regional language
6. **Episode Timeout**: Max 4 steps, then forced termination with 0.0 reward

---

## 🏗️ Project Structure

```
medbridge/
├── models.py              # Action / Observation schemas (2-step)
├── client.py              # WebSocket client for training
├── patients.py            # Random patient profile generator
├── reports.py             # 20 medical report templates
├── followups.py           # Follow-up Q&A rules (multilingual)
├── rewards.py             # 5 independent reward functions
├── env.py                 # Standalone environment class
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── Dockerfile             # Container build
└── server/
    ├── medbridge_environment.py  # OpenEnv Environment (main)
    └── app.py                    # FastAPI server
```

---

## 📈 Training

Training is done via **GRPO (Group Relative Policy Optimization)** using **TRL + Unsloth** on a T4 GPU.

→ [Training Notebook (Colab)](TODO_LINK)

---

## 🤝 Why This Matters

- **800M Indians** lack access to medical translation
- **No existing RL environment** for multilingual medical communication
- **Verifiable rewards** — all 5 dimensions are programmatically checkable
- **Real-world impact** — trained model could power WhatsApp health bots, hospital kiosks, and telemedicine apps

---

## 📝 License

This project is built on OpenEnv (BSD License).
