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

# 🏥 MedBridge — Medical Communication RL Environment

A 2-step RL environment that trains LLMs to explain complex medical diagnoses to Indian patients in their native language.

**→ See [full README](https://github.com/Aries0x/Medbridge) for details.**

## Quick Start

```python
from medbridge import MedbridgeAction, MedbridgeEnv

with MedbridgeEnv(base_url="https://Nermal007-medbridge.hf.space").sync() as env:
    result = env.reset()
    obs = result.observation
    print(f"Patient: {obs.patient_profile['name']}, {obs.patient_profile['language']}")
    print(f"Diagnosis: {obs.diagnosis_name} ({obs.severity})")

    result = env.step(MedbridgeAction(
        explanation="Your blood sugar is very high. You need medicine daily."
    ))
    print(f"Follow-up: {result.observation.followup_question}")
```

## Links

- [GitHub](https://github.com/Aries0x/Medbridge)
- [Training Notebook](https://colab.research.google.com/drive/1r2JGgFEcvQidzyksnTtBwSNUV_KFG0_0?usp=sharing)
- [Blog Post](https://github.com/Aries0x/Medbridge/blob/main/Medbridge_Blog.md)
