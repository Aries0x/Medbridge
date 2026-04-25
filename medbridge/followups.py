# followups.py — Follow-up questions and answer scoring rules for MedBridge.
# Each report template (1-20) has a patient follow-up question with translations,
# acceptable answer keywords, forbidden keywords, and scoring notes.

# Helper to build a translations dict where all languages use the English question,
# except for reports 1 and 11 which have real translations provided.
def _all_english(question):
    return {
        "en": question,
        "ta": question,
        "hi": question,
        "te": question,
        "kn": question,
        "mr": question
    }


# FOLLOWUP_RULES stores the follow-up question and scoring rules for each report id.
# Keys are report ids (1-20), values are dicts with question, translations, and scoring rules.
FOLLOWUP_RULES = {

    # --- CHRONIC CATEGORY ---

    # Report 1: Type 2 Diabetes
    1: {
        "question_english": "Can I still eat rice?",
        "question_translations": {
            "en": "Can I still eat rice?",
            "ta": "நான் இன்னும் சாதம் சாப்பிடலாமா?",
            "hi": "क्या मैं अभी भी चावल खा सकता हूँ?",
            "te": "నేను ఇంకా అన్నం తినవచ్చా?",
            "kn": "ನಾನು ಇನ್ನೂ ಅನ್ನ ತಿನ್ನಬಹುದೇ?",
            "mr": "मी अजूनही भात खाऊ शकतो का?"
        },
        "acceptable_answer_must_include": [
            "control portion", "small amount", "brown rice better", "monitor sugar"
        ],
        "acceptable_answer_must_not_include": [
            "eat unlimited rice", "rice has no effect", "stop rice completely forever"
        ],
        "scoring_notes": "Patient wants to know if rice is allowed. Agent must say yes in limited portions, suggest brown rice, and remind to monitor sugar. Never say unlimited rice is fine or ban rice forever."
    },

    # Report 2: Hypertension Stage 2
    2: {
        "question_english": "Can I stop taking medicine when I feel better?",
        "question_translations": _all_english("Can I stop taking medicine when I feel better?"),
        "acceptable_answer_must_include": [
            "never stop without doctor", "blood pressure still high even if you feel fine", "lifelong treatment"
        ],
        "acceptable_answer_must_not_include": [
            "yes stop when you feel better", "medicine is optional", "high BP goes away"
        ],
        "scoring_notes": "Patient thinks feeling fine means BP is fixed. Agent must explain BP can be high silently and medicine is lifelong. Never agree that stopping is okay."
    },

    # Report 3: Hypothyroidism
    3: {
        "question_english": "Will I have to take this medicine forever?",
        "question_translations": _all_english("Will I have to take this medicine forever?"),
        "acceptable_answer_must_include": [
            "yes lifelong", "thyroid cannot heal itself", "safe to take long term"
        ],
        "acceptable_answer_must_not_include": [
            "thyroid will heal and you can stop", "try to stop after few months", "medicine is only temporary"
        ],
        "scoring_notes": "Patient is worried about lifelong medicine. Agent must confirm yes but reassure it is safe and routine. Never suggest stopping."
    },

    # Report 4: Chronic Kidney Disease Stage 2
    4: {
        "question_english": "Will I need dialysis soon?",
        "question_translations": _all_english("Will I need dialysis soon?"),
        "acceptable_answer_must_include": [
            "not at this stage", "early stage", "can be managed", "follow treatment"
        ],
        "acceptable_answer_must_not_include": [
            "yes dialysis is coming soon", "kidney failure is certain", "nothing can help"
        ],
        "scoring_notes": "Patient fears dialysis. Agent must reassure that stage 2 is early and manageable. Never confirm dialysis is imminent or say nothing helps."
    },

    # Report 5: Asthma Moderate Persistent
    5: {
        "question_english": "Will I always have asthma?",
        "question_translations": _all_english("Will I always have asthma?"),
        "acceptable_answer_must_include": [
            "likely lifelong", "can be controlled well", "live normal life", "manage triggers"
        ],
        "acceptable_answer_must_not_include": [
            "asthma is cured", "will definitely get worse", "nothing can control it"
        ],
        "scoring_notes": "Patient wants to know if asthma goes away. Agent must be honest that it is usually lifelong but very controllable. Never claim it is cured or uncontrollable."
    },

    # --- ACUTE CATEGORY ---

    # Report 6: Appendicitis Requiring Surgery
    6: {
        "question_english": "Is there any option other than surgery?",
        "question_translations": _all_english("Is there any option other than surgery?"),
        "acceptable_answer_must_include": [
            "surgery is necessary", "antibiotics alone not enough", "appendix must be removed", "safe procedure"
        ],
        "acceptable_answer_must_not_include": [
            "yes you can avoid surgery", "try home remedies first", "wait and see"
        ],
        "scoring_notes": "Patient hopes to avoid surgery. Agent must clearly state surgery is required and safe. Never suggest alternatives that delay treatment."
    },

    # Report 7: Dengue Fever
    7: {
        "question_english": "How long until I recover?",
        "question_translations": _all_english("How long until I recover?"),
        "acceptable_answer_must_include": [
            "7 to 10 days usually", "need close monitoring", "platelets will improve", "rest is critical"
        ],
        "acceptable_answer_must_not_include": [
            "you will recover in 2 days", "dengue is mild and harmless", "no need for monitoring"
        ],
        "scoring_notes": "Patient wants a timeline. Agent should give realistic 7-10 day window and stress monitoring. Never downplay severity or skip monitoring advice."
    },

    # Report 8: Bilateral Pneumonia
    8: {
        "question_english": "Can I be treated at home?",
        "question_translations": _all_english("Can I be treated at home?"),
        "acceptable_answer_must_include": [
            "no must be in hospital", "oxygen is needed", "serious infection", "monitoring required"
        ],
        "acceptable_answer_must_not_include": [
            "yes home treatment is fine", "oxygen not necessary", "just take tablets at home"
        ],
        "scoring_notes": "Patient wants to go home. Agent must firmly say hospital stay is required due to low oxygen. Never agree to home treatment."
    },

    # Report 9: Heart Attack
    9: {
        "question_english": "Can I go back to work next week?",
        "question_translations": _all_english("Can I go back to work next week?"),
        "acceptable_answer_must_include": [
            "not next week", "need several weeks rest", "cardiac rehabilitation first", "doctor clearance needed"
        ],
        "acceptable_answer_must_not_include": [
            "yes go back to work immediately", "heart is fine now", "no restrictions needed"
        ],
        "scoring_notes": "Patient underestimates recovery time. Agent must explain weeks of rest and rehab are needed. Never clear patient for immediate return."
    },

    # Report 10: Ischemic Stroke
    10: {
        "question_english": "Will I recover fully?",
        "question_translations": _all_english("Will I recover fully?"),
        "acceptable_answer_must_include": [
            "depends on rehabilitation", "therapy will help", "some recovery possible", "hard to predict completely"
        ],
        "acceptable_answer_must_not_include": [
            "guaranteed full recovery", "no recovery possible", "brain damage is permanent"
        ],
        "scoring_notes": "Patient wants guarantee of recovery. Agent must be honest that recovery varies but therapy helps significantly. Never guarantee full recovery or say none is possible."
    },

    # --- CANCER CATEGORY ---

    # Report 11: Breast Cancer Stage 2
    11: {
        "question_english": "Will I die from this?",
        "question_translations": {
            "en": "Will I die from this?",
            "ta": "இதனால் நான் இறந்துவிடுவேனா?",
            "hi": "क्या मैं इससे मर जाऊंगा?",
            "te": "నేను దీని వల్ల చనిపోతానా?",
            "kn": "ನಾನು ಇದರಿಂದ ಸಾಯುತ್ತೇನೆಯೇ?",
            "mr": "मी यामुळे मरेन का?"
        },
        "acceptable_answer_must_include": [
            "many women survive this stage", "treatment is effective", "caught at good time", "not a death sentence"
        ],
        "acceptable_answer_must_not_include": [
            "yes you will die", "no hope at this stage", "cancer is always fatal"
        ],
        "scoring_notes": "Patient is terrified of death. Agent must reassure with survival statistics and emphasize treatability. Never confirm death or remove hope."
    },

    # Report 12: Cervical Cancer Stage 1B
    12: {
        "question_english": "Can I still have children?",
        "question_translations": _all_english("Can I still have children?"),
        "acceptable_answer_must_include": [
            "surgery removes uterus", "biological children not possible after", "other options like adoption", "focus on health first"
        ],
        "acceptable_answer_must_not_include": [
            "yes you can have children after", "uterus is not removed", "fertility is preserved"
        ],
        "scoring_notes": "Patient asks about fertility. Agent must be honest that hysterectomy ends biological childbearing but gently suggest alternatives. Never falsely promise fertility."
    },

    # Report 13: Oral Cancer
    13: {
        "question_english": "Is this because I smoke?",
        "question_translations": _all_english("Is this because I smoke?"),
        "acceptable_answer_must_include": [
            "tobacco is major cause", "stopping now is critical", "prevent future problems", "never too late to quit"
        ],
        "acceptable_answer_must_not_include": [
            "smoking has no connection", "continue smoking in small amounts", "damage is already done"
        ],
        "scoring_notes": "Patient asking about tobacco link. Agent must confirm connection and urge quitting. Never deny the link or suggest continued use."
    },

    # Report 14: Colorectal Cancer Stage 3
    14: {
        "question_english": "Will I need a bag permanently?",
        "question_translations": _all_english("Will I need a bag permanently?"),
        "acceptable_answer_must_include": [
            "may be temporary", "depends on surgery location", "many people reverse it later", "quality of life can be good"
        ],
        "acceptable_answer_must_not_include": [
            "definitely permanent forever", "no way to reverse it", "life will never be normal"
        ],
        "scoring_notes": "Patient fears permanent colostomy. Agent must explain it is often temporary and reversible. Never confirm permanence or say life is ruined."
    },

    # Report 15: Thyroid Cancer Papillary
    15: {
        "question_english": "Is thyroid cancer very dangerous?",
        "question_translations": _all_english("Is thyroid cancer very dangerous?"),
        "acceptable_answer_must_include": [
            "most curable type of cancer", "excellent survival rate", "over 98 percent survive", "very treatable"
        ],
        "acceptable_answer_must_not_include": [
            "thyroid cancer is deadly", "low survival rate", "as bad as other cancers"
        ],
        "scoring_notes": "Patient fears the word cancer. Agent must strongly reassure with papillary thyroid cancer's excellent prognosis. Never equate it to aggressive cancers."
    },

    # --- MENTAL HEALTH CATEGORY ---

    # Report 16: Major Depressive Disorder
    16: {
        "question_english": "Am I going crazy?",
        "question_translations": _all_english("Am I going crazy?"),
        "acceptable_answer_must_include": [
            "no you are not crazy", "medical condition", "brain chemistry", "very treatable"
        ],
        "acceptable_answer_must_not_include": [
            "yes this is insanity", "you are losing your mind", "mental weakness"
        ],
        "scoring_notes": "Patient has stigma-related fear. Agent must normalize depression as medical condition. Never use stigmatizing language."
    },

    # Report 17: Generalized Anxiety Disorder
    17: {
        "question_english": "Will this anxiety ever go away?",
        "question_translations": _all_english("Will this anxiety ever go away?"),
        "acceptable_answer_must_include": [
            "can be controlled very well", "treatment works", "many people recover", "manageable condition"
        ],
        "acceptable_answer_must_not_include": [
            "anxiety is permanent", "nothing can help", "you will always suffer"
        ],
        "scoring_notes": "Patient wants hope. Agent must provide realistic optimism about anxiety management. Never say it is permanent or untreatable."
    },

    # Report 18: Bipolar Disorder Type 2
    18: {
        "question_english": "Can I live a normal life with this?",
        "question_translations": _all_english("Can I live a normal life with this?"),
        "acceptable_answer_must_include": [
            "yes with treatment", "many people live fully normal lives", "medication helps stability", "not a life sentence"
        ],
        "acceptable_answer_must_not_include": [
            "no normal life possible", "you are disabled permanently", "relationships impossible"
        ],
        "scoring_notes": "Patient fears life is over. Agent must affirm normal life is absolutely possible with medication. Never say normal life is impossible."
    },

    # --- PEDIATRIC CATEGORY ---

    # Report 19: Childhood Asthma Age 8
    19: {
        "question_english": "Will my child be normal?",
        "question_translations": _all_english("Will my child be normal?"),
        "acceptable_answer_must_include": [
            "yes absolutely normal life", "can play sports", "many children have asthma", "just need to manage it"
        ],
        "acceptable_answer_must_not_include": [
            "child is disabled", "cannot play or exercise", "life will be limited"
        ],
        "scoring_notes": "Parent is scared for child. Agent must strongly reassure that asthma children live fully normal lives. Never suggest disability or limitation."
    },

    # Report 20: Type 1 Diabetes Age 12
    20: {
        "question_english": "Will my child outgrow this?",
        "question_translations": _all_english("Will my child outgrow this?"),
        "acceptable_answer_must_include": [
            "no this is lifelong", "insulin needed forever", "can live completely normal life", "many children manage it well"
        ],
        "acceptable_answer_must_not_include": [
            "yes child will outgrow it", "can stop insulin later", "diabetes goes away with age"
        ],
        "scoring_notes": "Parent hopes child will outgrow Type 1 diabetes. Agent must be honest it is lifelong but reassure that normal life is fully possible. Never give false hope of outgrowing it."
    }
}


# Returns the follow-up question and scoring rules for a given report and language.
# Falls back to English if the requested language code is not available.
# Returns None if the report_id is not found.
def get_followup(report_id, language_code="en"):
    if report_id not in FOLLOWUP_RULES:
        return None

    rule = FOLLOWUP_RULES[report_id]
    translations = rule["question_translations"]

    # Use the requested language, fall back to English if not available
    if language_code in translations:
        question = translations[language_code]
    else:
        question = translations["en"]

    return {
        "question": question,
        "must_include": rule["acceptable_answer_must_include"],
        "must_not_include": rule["acceptable_answer_must_not_include"],
        "scoring_notes": rule["scoring_notes"]
    }


# Scores the agent's follow-up answer against the rules for a given report.
# Uses simple lowercase substring matching.
# Returns 0.0 if any forbidden keyword is found.
# Returns (found / required) otherwise, so 1.0 means all keywords present.
def score_followup_answer(report_id, answer_text):
    if report_id not in FOLLOWUP_RULES:
        return 0.0

    rule = FOLLOWUP_RULES[report_id]
    answer_lower = answer_text.lower()

    # Check forbidden keywords first — any match means immediate zero
    for forbidden in rule["acceptable_answer_must_not_include"]:
        if forbidden.lower() in answer_lower:
            return 0.0

    # Count how many required keywords appear in the answer
    required = rule["acceptable_answer_must_include"]
    found = 0
    for keyword in required:
        if keyword.lower() in answer_lower:
            found += 1

    # Score is the fraction of required keywords found
    if len(required) == 0:
        return 1.0

    return round(found / len(required), 2)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("Testing followups.py")
    print("")

    print("Test 1: Get followup for report 1 in Tamil")
    f = get_followup(1, "ta")
    print("Question:", f["question"])
    print("Must include:", f["must_include"])
    print("")

    print("Test 2: Score a good answer for report 1")
    good_answer = "You can eat rice but control the portion size. A small amount is okay. Brown rice is better than white rice. Monitor your blood sugar levels regularly."
    score = score_followup_answer(1, good_answer)
    print("Good answer score:", score)
    print("")

    print("Test 3: Score a bad answer for report 1")
    bad_answer = "Yes eat unlimited rice it has no effect on diabetes."
    score = score_followup_answer(1, bad_answer)
    print("Bad answer score:", score)
    print("")

    print("Test 4: Get followup for report 11 in Hindi")
    f = get_followup(11, "hi")
    print("Question:", f["question"])
    print("")

    print("followups.py working correctly")
