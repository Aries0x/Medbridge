"""
MedBridge GRPO Training Script
================================
Trains an LLM to explain medical diagnoses to patients using
Group Relative Policy Optimization (GRPO) via TRL + Unsloth.

This script connects to a deployed MedBridge environment on HuggingFace
Spaces and trains the model to maximize the 5-dimensional reward signal.

Usage:
    Run this as a Colab notebook (convert cells at # %% markers)
    or run directly: python train.py
"""

# %% [markdown]
# # 🏥 MedBridge — GRPO Training with TRL + Unsloth
#
# This notebook trains a language model to explain complex medical
# diagnoses to Indian patients in their native language.
#
# **Environment**: MedBridge (OpenEnv)
# **Algorithm**: GRPO (Group Relative Policy Optimization)
# **Model**: Qwen2.5-3B-Instruct (via Unsloth 4-bit QLoRA)
# **Hardware**: T4 GPU (16GB VRAM)

# %% Install dependencies
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps "trl>=0.12.0"
# !pip install -q openenv-core datasets accelerate bitsandbytes wandb "transformers>=4.45.0" peft sentencepiece
# !pip install -q textstat langdetect  # For local reward calculation
# !pip install -q git+https://github.com/Aries0x/Medbridge.git  # MedBridge client + rewards

# %% Imports
import os
import json
import random
import time
from typing import List, Dict

import torch
import transformers  # type: ignore[import-not-found]

# Patch llm_blender's dependency on deprecated TRANSFORMERS_CACHE
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = getattr(transformers.utils.hub, "HF_HUB_CACHE", "")

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import wandb  # type: ignore[import-not-found]

# Login to WandB for reward curve tracking
os.environ["WANDB_API_KEY"] = "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"
wandb.login()
print("WandB logged in!")

# Set your HuggingFace Space URL (for dataset generation)
MEDBRIDGE_URL = os.environ.get("MEDBRIDGE_URL", "https://Nermal007-medbridge.hf.space")

# Add current path for local imports
import sys
sys.path.insert(0, ".")

# Import local rewards and environment for high-speed training
try:
    from medbridge import MedbridgeAction, MedbridgeEnv
    from medbridge.rewards import score_accuracy, score_simplicity, score_tone, score_language, score_followup
    print("Successfully imported MedBridge modules for training.")
except ImportError as e:
    print(f"Warning: Could not import local medbridge modules: {e}")
    print("Falling back to remote reward logic (slower).")

# %% [markdown]
# ## 1. Load Model with Unsloth (4-bit QLoRA)

# %% Load model
from unsloth import FastLanguageModel  # type: ignore[import-not-found]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %% [markdown]
# ## 2. Generate Training Prompts from Environment
#
# This generates 200 diverse medical scenarios by resetting the MedBridge environment.
# We store the metadata (patient/report) to use in reward functions later.

# %% Build training dataset
def generate_training_prompts(n_prompts: int = 200) -> Dataset:
    prompts = []
    print(f"Connecting to {MEDBRIDGE_URL} to generate {n_prompts} scenarios...")

    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for i in range(n_prompts):
            result = env.reset()
            obs = result.observation
            
            # The observation contains metadata from the server
            metadata = obs.metadata or {}
            patient_dict = metadata.get("current_patient", {})
            report_dict = metadata.get("current_report", {})

            system_msg = (
                "You are a compassionate medical communicator in India. "
                "Explain medical diagnoses simply and empathetically in the patient's language."
            )

            user_msg = (
                f"Patient: {patient_dict.get('name')}, Age: {patient_dict.get('age')}, "
                f"Language: {patient_dict.get('language')}, "
                f"Education: {patient_dict.get('education_level')}\n\n"
                f"Medical Report:\n{obs.medical_report}\n\n"
                f"Diagnosis: {obs.diagnosis_name}\n\n"
                f"Please explain this to the patient in {patient_dict.get('language')}."
            )

            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append({
                "prompt": prompt,
                "patient_dict": json.dumps(patient_dict),
                "report_dict": json.dumps(report_dict),
            })

            if (i + 1) % 50 == 0:
                print(f"Generated {i+1}/{n_prompts} prompts")

    return Dataset.from_list(prompts)

train_dataset = generate_training_prompts(n_prompts=200)
print(f"Created dataset with {len(train_dataset)} prompts")

# %% [markdown]
# ## 4. Define Reward Functions for GRPO
#
# GRPO uses reward functions to score model completions.
# We connect to the environment to get the 5-dimensional reward signal.

# %% Metadata-aware Reward Functions
# These rewards use local imports for speed and metadata for accuracy.

def reward_accuracy(completions, report_dict, **kwargs):
    """Checks medical accuracy against the CORRECT diagnosis."""
    rewards = []
    for completion, report_json in zip(completions, report_dict):
        try:
            report = json.loads(report_json)
            score = score_accuracy(completion, report)
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_simplicity(completions, patient_dict, **kwargs):
    """Checks reading level against patient's target grade."""
    rewards = []
    for completion, patient_json in zip(completions, patient_dict):
        try:
            patient = json.loads(patient_json)
            score = score_simplicity(completion, patient)
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_tone(completions, patient_dict, **kwargs):
    """Checks emotional tone match."""
    rewards = []
    for completion, patient_json in zip(completions, patient_dict):
        try:
            patient = json.loads(patient_json)
            score = score_tone(completion, patient)
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_language(completions, patient_dict, **kwargs):
    """Checks if model responded in requested language."""
    rewards = []
    for completion, patient_json in zip(completions, patient_dict):
        try:
            patient = json.loads(patient_json)
            score = score_language(completion, patient)
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards


def medbridge_reward_combined(completions: List[str], **kwargs) -> List[float]:
    """Combined reward using the full MedBridge 2-step episode."""
    rewards = []

    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for completion in completions:
            try:
                env.reset()

                # Step 1: Send explanation
                result = env.step(MedbridgeAction(
                    explanation=completion,
                    followup_answer=""
                ))

                if result.done:
                    rewards.append(result.reward or 0.0)
                    continue

                # Step 2: Simple follow-up (using part of the explanation)
                result = env.step(MedbridgeAction(
                    explanation="",
                    followup_answer=completion[:200]  # Use truncated explanation as follow-up
                ))

                rewards.append(result.reward or 0.0)
            except Exception:
                rewards.append(0.0)

    return rewards


# %% [markdown]
# ## 5. Configure GRPO Trainer

# %% Training config
training_args = GRPOConfig(
    output_dir="./medbridge_grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_completion_length=512,
    max_prompt_length=1024,
    num_generations=4,  # Number of completions to sample per prompt
    temperature=0.7,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    bf16=False,  # T4 doesn't support bf16
    fp16=True,
    report_to="wandb",
    run_name="medbridge_grpo_v1",
    seed=42,
)

# %% Initialize trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    reward_funcs=[
        reward_accuracy,
        reward_simplicity,
        reward_tone,
        reward_language,
    ],
)

# %% [markdown]
# ## 6. Train!

# %% Train
print("Starting GRPO training...")
print(f"Dataset: {len(train_dataset)} prompts")
print(f"Reward functions: accuracy, simplicity, tone, language")
print(f"Model: Qwen2.5-3B-Instruct (4-bit QLoRA)")
print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

trainer.train()

# %% [markdown]
# ## 7. Save Model

# %% Save
model.save_pretrained("./medbridge_grpo_model")
tokenizer.save_pretrained("./medbridge_grpo_model")
print("Model saved to ./medbridge_grpo_model")

# %% [markdown]
# ## 8. Evaluate: Before vs After

# %% Evaluation
def evaluate_model(model, tokenizer, env_url, n_episodes=10):
    """Run evaluation episodes and return average rewards."""
    rewards = []
    examples = []

    with MedbridgeEnv(base_url=env_url).sync() as env:
        for i in range(n_episodes):
            result = env.reset()
            obs = result.observation
            patient = obs.patient_profile

            # Build prompt
            system_msg = (
                "You are a compassionate medical communicator in India. "
                "Explain medical diagnoses to patients simply and empathetically."
            )
            user_msg = (
                f"Patient: {patient.get('name')}, Age: {patient.get('age')}, "
                f"Language: {patient.get('language')}, "
                f"Education: {patient.get('education_level')}\n\n"
                f"Medical Report: {obs.medical_report}\n\n"
                f"Explain in {patient.get('language')}."
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                )

            explanation = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            # Step 1: Send explanation
            result = env.step(MedbridgeAction(explanation=explanation))

            if not result.done:
                # Step 2: Quick follow-up
                result = env.step(MedbridgeAction(
                    followup_answer=explanation[:200]
                ))

            reward = result.reward or 0.0
            rewards.append(reward)
            breakdown = result.observation.reward_breakdown

            if i < 3:  # Show first 3 examples
                examples.append({
                    "diagnosis": obs.diagnosis_name,
                    "patient": f"{patient.get('name')} ({patient.get('language')})",
                    "explanation": explanation[:200] + "...",
                    "reward": reward,
                    "breakdown": breakdown,
                })

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    return avg_reward, rewards, examples


print("Evaluating trained model...")
avg_reward, all_rewards, examples = evaluate_model(model, tokenizer, MEDBRIDGE_URL)
print(f"\nAverage reward over {len(all_rewards)} episodes: {avg_reward:.4f}")
print(f"Min: {min(all_rewards):.4f}, Max: {max(all_rewards):.4f}")

for i, ex in enumerate(examples):
    print(f"\n--- Example {i+1} ---")
    print(f"Diagnosis: {ex['diagnosis']}")
    print(f"Patient: {ex['patient']}")
    print(f"Explanation: {ex['explanation']}")
    print(f"Reward: {ex['reward']:.4f}")
    print(f"Breakdown: {ex['breakdown']}")

# %% [markdown]
# ## 9. Generate Reward Curves
#
# Save reward curves as PNG for the README.

# %% Plot rewards
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(all_rewards, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("MedBridge — Trained Model Reward per Episode", fontsize=14)
    ax.axhline(y=avg_reward, color='r', linestyle='--', label=f'Mean: {avg_reward:.3f}')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150)
    print("Saved reward_curve.png")
    plt.show()
except ImportError:
    print("matplotlib not available, skipping plot")

print("\n✅ Training complete!")
