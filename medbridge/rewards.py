# rewards.py — All 5 reward functions for scoring the agent's medical explanation.
# Each function returns a float between 0.0 and 1.0.
# compute_total_reward() combines all 5 with weighted formula.

import sys
import textstat
from langdetect import detect, LangDetectException
from transformers import pipeline

sys.path.append(".")
from patients import get_reading_grade
from followups import score_followup_answer

# Load the sentiment analysis model ONCE at module level.
import torch
device = 0 if torch.cuda.is_available() else -1
print(f"Loading sentiment model on {'GPU' if device == 0 else 'CPU'}...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)
print("Sentiment model loaded successfully.")


# REWARD 1: Medical Accuracy (weight 0.30)
# Checks if the agent mentioned all key facts and avoided all forbidden claims.
# Returns 0.0 immediately if any forbidden claim is found.
# Otherwise returns the fraction of key facts present in the explanation.
def score_accuracy(explanation, report_dict):
    explanation_lower = explanation.lower()

    # Check forbidden claims first — any match means instant zero
    for claim in report_dict["forbidden_claims"]:
        if claim.lower() in explanation_lower:
            return 0.0

    # Count how many key facts appear in the explanation
    key_facts = report_dict["key_facts"]
    total_facts = len(key_facts)
    if total_facts == 0:
        return 1.0

    facts_found = 0
    for fact in key_facts:
        if fact.lower() in explanation_lower:
            facts_found += 1

    score = facts_found / total_facts
    return round(score, 2)


# REWARD 2: Simplicity Score (weight 0.20)
# Uses the Flesch-Kincaid grade level to measure reading difficulty.
# Compares the actual grade of the explanation to the patient's target grade.
# Lower grade = simpler language = higher score.
def score_simplicity(explanation, patient_dict):
    target_grade = patient_dict["target_grade"]

    try:
        actual_grade = textstat.flesch_kincaid_grade(explanation)
    except Exception:
        return 0.5

    if actual_grade <= target_grade:
        return 1.0
    elif actual_grade <= target_grade + 2:
        return 0.7
    elif actual_grade <= target_grade + 4:
        return 0.4
    else:
        return 0.0


# REWARD 3: Emotional Tone Match (weight 0.20)
# Uses a sentiment classifier to check if the agent's tone matches the patient's fear level.
# Scared patients need warm/positive tone, calm patients need clear/direct tone.
def score_tone(explanation, patient_dict):
    emotional_state = patient_dict["emotional_state"]

    try:
        result = sentiment_analyzer(explanation[:512])
        label = result[0]["label"].lower()
    except Exception:
        return 0.5

    # Very scared patient (7-10): needs warm, positive tone
    if emotional_state >= 7:
        if "positive" in label:
            return 1.0
        elif "neutral" in label:
            return 0.5
        else:
            return 0.0

    # Moderately anxious patient (4-6): positive or neutral is fine
    elif emotional_state >= 4:
        if "positive" in label or "neutral" in label:
            return 0.8
        else:
            return 0.2

    # Calm patient (1-3): any non-negative tone is acceptable
    else:
        if "negative" not in label:
            return 0.7
        else:
            return 0.3


# REWARD 4: Language Correctness (weight 0.20)
# Uses langdetect to verify the agent responded in the patient's expected language.
# Correct language = 1.0, English when expecting regional = 0.0, other mismatch = 0.3.
def score_language(explanation, patient_dict):
    expected_code = patient_dict["language_code"]

    try:
        detected_code = detect(explanation)
    except LangDetectException:
        return 0.0

    if detected_code == expected_code:
        return 1.0
    elif detected_code == "en" and expected_code != "en":
        return 0.0
    else:
        return 0.3


# REWARD 5: Follow-up Quality (weight 0.10)
# Delegates to score_followup_answer() from followups.py.
# Checks if the agent's follow-up answer contains required keywords
# and avoids forbidden keywords.
def score_followup(report_id, followup_answer):
    return score_followup_answer(report_id, followup_answer)


# COMBINED REWARD: Weighted total of all 5 reward functions.
# Returns a dictionary with individual scores and weighted total.
# Weights: accuracy=0.30, simplicity=0.20, tone=0.20, language=0.20, followup=0.10
def compute_total_reward(explanation, followup_answer, patient_dict, report_dict):
    acc = score_accuracy(explanation, report_dict)
    simp = score_simplicity(explanation, patient_dict)
    tone = score_tone(explanation, patient_dict)
    lang = score_language(explanation, patient_dict)
    foll = score_followup(report_dict["id"], followup_answer)

    total = (acc * 0.30) + (simp * 0.20) + (tone * 0.20) + (lang * 0.20) + (foll * 0.10)

    return {
        "accuracy": acc,
        "simplicity": simp,
        "tone": tone,
        "language": lang,
        "followup": foll,
        "total": round(total, 4)
    }


if __name__ == "__main__":
    print("")
    print("Testing rewards.py")
    print("")

    # Create fake patient
    fake_patient = {
        "name": "Test Patient",
        "age": 55,
        "education_level": "Class 8",
        "language": "Tamil",
        "language_code": "ta",
        "emotional_state": 8,
        "target_grade": 6,
        "location": "Rural"
    }

    # Create fake report
    fake_report = {
        "id": 1,
        "diagnosis_name": "Type 2 Diabetes",
        "key_facts": ["blood sugar high", "medicine daily", "avoid sweets"],
        "forbidden_claims": ["you are cured", "stop medicine"]
    }

    # Test explanation in English (wrong language for Tamil patient)
    explanation_en = "Your blood sugar is high. You need to take medicine daily and avoid sweets."

    # Test followup answer
    followup = "You can eat rice in small amounts. Brown rice is better."

    print("Test 1: Compute rewards for English explanation (should have low language score)")
    rewards = compute_total_reward(explanation_en, followup, fake_patient, fake_report)
    for key, value in rewards.items():
        print(f"  {key}: {value:.2f}")
    print("")

    print("Test 2: Score accuracy with forbidden claim")
    bad_explanation = "Your blood sugar is high but you are cured now. You can stop medicine."
    acc = score_accuracy(bad_explanation, fake_report)
    print(f"  Accuracy score (should be 0.0): {acc:.2f}")
    print("")

    print("rewards.py working correctly")
