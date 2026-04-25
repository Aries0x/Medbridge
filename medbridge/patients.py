import random

# List of 20 common Indian names used for generating patient profiles
PATIENT_NAMES = [
    "Lakshmi", "Priya", "Ravi", "Suresh",
    "Meena", "Kavitha", "Arjun", "Divya",
    "Rajesh", "Anitha", "Murugan", "Saranya",
    "Venkat", "Padma", "Karthik", "Nalini",
    "Shankar", "Geetha", "Balu", "Sumathi"
]

# Education levels representing typical Indian schooling milestones
EDUCATION_LEVELS = ["Class 5", "Class 8", "Class 10", "Graduate"]

# Regional languages commonly spoken across India
LANGUAGES = ["Tamil", "Hindi", "Telugu", "Kannada", "Marathi"]

# Location types representing Indian population distribution
LOCATIONS = ["Rural", "Semi-urban", "Urban"]

# Gender options
GENDERS = ["Male", "Female"]


# Returns an emotional label string based on the fear/anxiety score (1-10).
# Low scores mean the patient is calm, high scores mean extreme fear.
def get_emotional_label(emotional_state):
    if emotional_state <= 3:
        return "Calm"
    elif emotional_state <= 6:
        return "Moderately anxious"
    elif emotional_state <= 8:
        return "Scared"
    else:
        return "Extremely scared"


# Returns the target reading grade level based on the patient's education.
# Lower education means simpler language is needed in explanations.
def get_reading_grade(education_level):
    grade_map = {
        "Class 5": 4,
        "Class 8": 6,
        "Class 10": 8,
        "Graduate": 10
    }
    return grade_map.get(education_level, 6)


# Returns the langdetect language code for a given Indian language name.
# Used by the language reward function to verify the agent's response language.
def get_language_code(language):
    code_map = {
        "Tamil": "ta",
        "Hindi": "hi",
        "Telugu": "te",
        "Kannada": "kn",
        "Marathi": "mr",
        "English": "en"
    }
    return code_map.get(language, "en")


# Generates a complete random patient profile dictionary.
# Picks random values for all demographic fields and computes
# derived fields (emotional_label, target_grade, language_code)
# automatically so they are ready for the environment to use.
def generate_patient():
    name = random.choice(PATIENT_NAMES)
    age = random.randint(30, 75)
    education_level = random.choice(EDUCATION_LEVELS)
    language = random.choice(LANGUAGES)
    emotional_state = random.randint(1, 10)
    location = random.choice(LOCATIONS)
    gender = random.choice(GENDERS)

    patient = {
        "name": name,
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "language": language,
        "emotional_state": emotional_state,
        "location": location,
        "emotional_label": get_emotional_label(emotional_state),
        "target_grade": get_reading_grade(education_level),
        "language_code": get_language_code(language)
    }

    return patient


if __name__ == "__main__":
    print("Testing patients.py")
    print("=" * 50)
    print("")

    for i in range(5):
        patient = generate_patient()
        print(f"Patient {i + 1}:")
        print(f"  Name:             {patient['name']}")
        print(f"  Age:              {patient['age']}")
        print(f"  Gender:           {patient['gender']}")
        print(f"  Education:        {patient['education_level']}")
        print(f"  Language:         {patient['language']}")
        print(f"  Location:         {patient['location']}")
        print(f"  Emotional State:  {patient['emotional_state']}/10 ({patient['emotional_label']})")
        print(f"  Target Grade:     {patient['target_grade']}")
        print(f"  Language Code:    {patient['language_code']}")
        print("")

    print("patients.py working correctly")
