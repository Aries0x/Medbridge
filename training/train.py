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
# !pip install -q unsloth openenv-core datasets accelerate bitsandbytes wandb
# !pip install -U "trl>=0.12.0" "transformers>=4.45.0" "mergekit" "llm_blender"

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

print(f"Model loaded. Trainable params: {model.print_trainable_parameters()}")

# %% [markdown]
# ## 2. Connect to MedBridge Environment

# %%  Environment setup
# Set your HuggingFace Space URL here
MEDBRIDGE_URL = os.environ.get("MEDBRIDGE_URL", "https://Nermal007-medbridge.hf.space")

# We'll use the OpenEnv client to interact with the environment
import sys
sys.path.insert(0, ".")

# For local testing, import directly
# For remote, install the client: pip install git+https://github.com/YOUR_REPO
try:
    from medbridge import MedbridgeAction, MedbridgeEnv
    print("Using local MedBridge client")
except ImportError:
    print("MedBridge client not found locally. Install from HF Space repo.")
    raise

# %% [markdown]
# ## 3. Define the Rollout Function
#
# This function:
# 1. Generates prompts from the environment (reset)
# 2. Gets model completions
# 3. Sends completions to the environment (step)
# 4. Returns rewards for GRPO

# %% Build training dataset from environment
def generate_training_prompts(n_prompts: int = 200) -> Dataset:
    """
    Generate training prompts by resetting the MedBridge environment.
    Each prompt contains a patient scenario that the model must respond to.
    """
    prompts = []

    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for i in range(n_prompts):
            result = env.reset()
            obs = result.observation

            patient = obs.patient_profile
            system_msg = (
                "You are a compassionate medical communicator in India. "
                "Your job is to explain medical diagnoses to patients in simple language "
                "they can understand, in their preferred language, matching their emotional needs."
            )

            user_msg = (
                f"Patient: {patient.get('name', 'Patient')}, "
                f"Age: {patient.get('age', 50)}, "
                f"Language: {patient.get('language', 'Hindi')}, "
                f"Education: {patient.get('education_level', 'Class 8')}, "
                f"Emotional state: {patient.get('emotional_label', 'Anxious')}\n\n"
                f"Medical Report:\n{obs.medical_report}\n\n"
                f"Diagnosis: {obs.diagnosis_name} (Severity: {obs.severity})\n\n"
                f"Please explain this diagnosis to the patient in {patient.get('language', 'Hindi')}. "
                f"Use simple words appropriate for {patient.get('education_level', 'Class 8')} education level. "
                f"Be empathetic and match their emotional state."
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
                "patient_language": patient.get("language", "Hindi"),
                "diagnosis_name": obs.diagnosis_name,
                "severity": obs.severity,
            })

            if (i + 1) % 50 == 0:
                print(f"Generated {i+1}/{n_prompts} prompts")

    dataset = Dataset.from_list(prompts)
    print(f"Dataset created: {len(dataset)} prompts")
    return dataset

# %% Generate dataset
print("Generating training prompts from MedBridge environment...")
train_dataset = generate_training_prompts(n_prompts=200)
print(train_dataset)

# %% [markdown]
# ## 4. Define Reward Functions for GRPO
#
# GRPO uses reward functions to score model completions.
# We connect to the environment to get the 5-dimensional reward signal.

# %% Reward functions
def medbridge_reward_accuracy(completions: List[str], **kwargs) -> List[float]:
    """Score accuracy of medical explanations via MedBridge environment."""
    rewards = []
    prompts = kwargs.get("prompts", [""] * len(completions))

    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for i, completion in enumerate(completions):
            try:
                result = env.reset()

                # Step 1: Send explanation
                result = env.step(MedbridgeAction(
                    explanation=completion,
                    followup_answer=""
                ))

                # Get accuracy score from partial rewards
                scores = result.observation.reward_breakdown
                rewards.append(scores.get("accuracy", 0.0))
            except Exception as e:
                rewards.append(0.0)

    return rewards


def medbridge_reward_simplicity(completions: List[str], **kwargs) -> List[float]:
    """Score simplicity of explanations."""
    rewards = []
    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for completion in completions:
            try:
                env.reset()
                result = env.step(MedbridgeAction(explanation=completion, followup_answer=""))
                scores = result.observation.reward_breakdown
                rewards.append(scores.get("simplicity", 0.0))
            except Exception:
                rewards.append(0.0)
    return rewards


def medbridge_reward_tone(completions: List[str], **kwargs) -> List[float]:
    """Score emotional tone."""
    rewards = []
    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for completion in completions:
            try:
                env.reset()
                result = env.step(MedbridgeAction(explanation=completion, followup_answer=""))
                scores = result.observation.reward_breakdown
                rewards.append(scores.get("tone", 0.0))
            except Exception:
                rewards.append(0.0)
    return rewards


def medbridge_reward_language(completions: List[str], **kwargs) -> List[float]:
    """Score language correctness."""
    rewards = []
    with MedbridgeEnv(base_url=MEDBRIDGE_URL).sync() as env:
        for completion in completions:
            try:
                env.reset()
                result = env.step(MedbridgeAction(explanation=completion, followup_answer=""))
                scores = result.observation.reward_breakdown
                rewards.append(scores.get("language", 0.0))
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
        medbridge_reward_accuracy,
        medbridge_reward_simplicity,
        medbridge_reward_tone,
        medbridge_reward_language,
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
