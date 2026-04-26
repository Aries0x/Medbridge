# MedBridge: From Medical Panic to Patient Clarity

If we had 30 seconds on stage, we would say this:

People are not only suffering from disease.
People are suffering from confusion.

A report is handed to a patient.
The words are technical.
The family is scared.
Nobody explains.

That silent gap between diagnosis and understanding is where MedBridge was born.

## The Story That Started It

One night, our friend Arjun called, voice shaking.
His father had received a liver report full of terms he could not understand. By the time Arjun called us, his father had already decided the worst was coming. He stopped eating. He stayed in bed. The whole house was in fear.

We read the report. The condition was serious, but manageable.
What was not manageable was the language.

A few months later, our friend Rohan called after an MRI.
He saw words like "disc bulge" and "annular tear," Googled them, and convinced himself he would not walk normally again.

Again, the diagnosis was treatable.
Again, the fear came from explanation failure.

Those two calls changed our direction.
We stopped asking, "Can AI summarize reports?"
We started asking, "Can AI explain reports the way a calm, caring human would?"

That question became MedBridge.

## The Problem We Are Solving

In many real healthcare settings, especially across rural and semi-urban populations, patients receive medical reports in complex English while they think and feel in regional languages.

When someone is already scared, medical jargon sounds like a threat.
In that moment, patients often do one of three things:

1. Panic and assume the worst
2. Ignore the report entirely
3. Follow unsafe advice from unverified sources

None of these outcomes help treatment.

So our mission became simple and human:
Translate fear into understanding.
Translate complexity into action.

## What MedBridge Actually Does

MedBridge is a reinforcement learning environment that trains a language model to explain the same diagnosis differently for different people.

Not differently in facts.
Differently in communication.

The model learns to adapt based on:

1. Language preference
2. Education level
3. Emotional state
4. Medical accuracy requirements
5. Follow-up question quality

In short, MedBridge trains for patient understanding, not just text generation.

## How It Works (Easy to Follow)

Every episode in MedBridge follows the same flow:

1. Generate a patient profile
Example: Lakshmi, age 58, Tamil speaker, Class 8 education, fear level 9/10.

2. Attach a medical report
Example: A breast cancer pathology report written in technical clinical language.

3. Ask the model to explain it
Task: explain truthfully, clearly, and compassionately in the patient's language.

4. Score the response using five independent judges

5. Reinforce better behavior and repeat

At first, the model sounds like a textbook.
After training, it sounds like guidance.

## Our Architecture Diagram

![MedBridge architecture diagram](screenshot_proofs/Architecture%20diagram.png)

This loop is why the model improves behavior over time, not just wording.

## Datasets and Scenario Design Used

To keep the environment realistic, we built a structured scenario bank:

1. 20 medical report templates across 5 categories
2. 6 languages: Tamil, Hindi, Telugu, Kannada, Marathi, English
3. 4 education levels: Class 5, Class 8, Class 10, Graduate
4. 10-point emotional scale: calm to extremely scared

### Medical categories included

1. Chronic: diabetes, hypertension, thyroid, kidney, asthma
2. Acute: appendicitis, dengue, pneumonia, heart attack, stroke
3. Cancer: breast, cervical, oral, colorectal, thyroid
4. Mental health: depression, anxiety, bipolar
5. Pediatric: childhood asthma, type 1 diabetes

Total practical combinations evaluated: 1,200+.

## The Five Judges (Reward Design)

This is the technical heart of MedBridge and the reason it does not collapse into "nice sounding but unsafe" responses.

1. Medical Accuracy (30%)
Checks whether key facts are preserved and unsafe claims are avoided.

2. Simplicity (20%)
Checks readability relative to patient education level.

3. Emotional Tone (20%)
Checks if response tone matches patient fear level.

4. Language Correctness (20%)
Checks if output is in the requested language.

5. Follow-up Handling (10%)
Checks whether hard questions are answered honestly and calmly.

One reward is easy to game.
Five independent rewards are much harder to game.
That is exactly what we wanted.

## What Changed After Training

In our benchmark scenario:

1. Initial reward: 0.21/1.00
2. After training: 0.81/1.00

But the bigger change is human, not numeric.

Before training, the model speaks in formal clinical text that increases anxiety.
After training, it preserves medical truth while giving patients a clear next step and emotional grounding.

That difference is the point of MedBridge.

## Training Graphs (Model Learning Evidence)

Below are the training screenshots captured from our runs:

![Training graph 1](screenshot_proofs/Screenshot%202026-04-26%20015341.png)

![Training graph 2](screenshot_proofs/Screenshot%202026-04-26%20015411.png)

![Training graph 3](screenshot_proofs/Screenshot%202026-04-26%20023138.png)

![Training graph 4](screenshot_proofs/Screenshot%202026-04-26%20023151.png)

## Stack We Used

1. OpenEnv for environment interface and reproducibility
2. TRL (GRPO) for reinforcement learning updates
3. Unsloth for memory-efficient training
4. Qwen2.5-3B-Instruct as base model
5. Hugging Face Spaces for demo hosting

We intentionally built this with accessible compute so students and small teams can replicate it.

## How We Will Present This Live

Our demo is built for non-technical judges.
In one run, we show:

1. A patient appears
2. A report appears
3. Untrained response appears
4. Five scores appear
5. Trained response appears
6. Score jump and response quality change become obvious

The audience does not need to know RL internals to feel the impact.
They only need to compare one moment:

Before: "I am scared and I do not understand."
After: "I understand what is happening, and I know what to do next."

## What We Learned

1. Communication is a clinical variable
If understanding fails, outcomes can fail.

2. Doctors need support, not replacement
Most clinicians want to explain better, but time is limited.

3. Human-centered constraints matter
Without multi-dimensional rewards, models drift toward shallow outputs.

4. Language is emotional infrastructure
People heal better when they are spoken to in words they trust.

## Why This Matters to Us Personally

MedBridge is not just a hackathon project for us.
It is a response to real calls from real friends, real fear, and real families sitting in uncertainty.

We are building this because no one should walk out of a clinic with a report in hand and terror in heart simply because the explanation failed.

If MedBridge helps even one family move from panic to plan, this project has already done something meaningful.

And if we scale it right, it can do far more.

## What Comes Next

1. Expand language coverage beyond current six
2. Grow scenario templates and safety rules
3. Add stronger clinician-in-the-loop validation
4. Pilot real deployment pathways with hospitals

The long-term goal is simple:
Every diagnosis should come with understanding.

## Closing Note

If this project resonates with you, we invite you to explore it end-to-end.
Not as a benchmark, but as a step toward more humane healthcare communication.

For us, MedBridge is about one promise:
No patient should leave with a diagnosis but without understanding.

## Reference Links

1. GitHub Repository: https://github.com/Aries0x/Medbridge
2. Google Colab Notebook: https://colab.research.google.com/drive/1r2JGgFEcvQidzyksnTtBwSNUV_KFG0_0?usp=sharing
3. Hugging Face Space: https://Nermal007-medbridge.hf.space

Thank you

Team Frost Byte