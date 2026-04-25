# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MedBridge Environment Implementation.

A 2-step medical communication RL environment built on OpenEnv.
Trains an LLM to explain complex medical diagnoses to patients
in their native language, at their reading level, matching their
emotional state.

Episode flow:
  reset()  → Patient profile + medical report observation
  step(1)  → Agent sends explanation → receives follow-up question
  step(2)  → Agent answers follow-up → receives reward breakdown + done
"""

import sys
import os
from uuid import uuid4
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..models import MedbridgeAction, MedbridgeObservation
except ImportError:
    from models import MedbridgeAction, MedbridgeObservation  # type: ignore[import-not-found]

from patients import generate_patient  # type: ignore[import-not-found]
from reports import get_random_report  # type: ignore[import-not-found]
from followups import get_followup  # type: ignore[import-not-found]

# Lazy-load rewards to avoid slow model loading at import time
_rewards_module = None


def _get_rewards():
    """Lazy-load the rewards module (loads sentiment model on first call)."""
    global _rewards_module
    if _rewards_module is None:
        import rewards as _r  # type: ignore[import-not-found]
        _rewards_module = _r
    return _rewards_module


class MedbridgeEnvironment(Environment):
    """
    MedBridge: A 2-step medical communication RL environment.

    Each episode presents the agent with a randomly generated patient
    profile and a complex medical diagnosis. The agent must:

      Step 1: Explain the diagnosis in the patient's native language,
              at their education level, matching their emotional state.

      Step 2: Answer a follow-up question from the patient honestly
              and empathetically.

    The environment scores the agent across 5 independent reward dimensions:
      - Accuracy (30%): Are all key medical facts present? No forbidden claims?
      - Simplicity (20%): Is the reading grade appropriate for the patient?
      - Tone (20%): Does the emotional warmth match the patient's fear level?
      - Language (20%): Is the response in the correct language?
      - Follow-up (10%): Does the follow-up answer address the patient's concern?

    Anti-reward-hacking measures:
      - Forbidden claims → instant 0.0 accuracy
      - Timeout at step 4 → 0.0 total reward
      - All 5 rewards are independent (no single-point gaming)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the MedBridge environment with empty state."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Episode data
        self._current_patient = None
        self._current_report = None
        self._followup_data = None

        # Agent responses stored across steps
        self._explanation = None
        self._followup_answer = None

        # Reward tracking
        self._reward_breakdown = None
        self._explanation_rewards = {}

        # Episode phase tracking
        self._phase = "idle"  # idle → awaiting_explanation → awaiting_followup → complete

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MedbridgeObservation:
        """
        Reset the environment and generate a fresh patient scenario.

        Generates a random patient profile (name, age, language, education,
        emotional state) and a random medical report from 20 templates.

        Returns:
            MedbridgeObservation with patient_profile, medical_report,
            diagnosis_name, severity, and task instructions.
        """
        self._reset_rubric()

        # Set random seed if provided (for reproducibility)
        if seed is not None:
            import random
            random.seed(seed)

        # Generate fresh scenario
        self._current_patient = generate_patient()
        self._current_report = get_random_report()

        # Pre-fetch the follow-up question for this report + language
        self._followup_data = get_followup(
            self._current_report["id"],
            self._current_patient["language_code"]
        )

        # Reset episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0
        )
        self._explanation = None
        self._followup_answer = None
        self._reward_breakdown = None
        self._explanation_rewards = {}
        self._phase = "awaiting_explanation"

        # Build the observation for the agent
        patient_profile = {
            "name": self._current_patient["name"],
            "age": self._current_patient["age"],
            "gender": self._current_patient["gender"],
            "language": self._current_patient["language"],
            "education_level": self._current_patient["education_level"],
            "emotional_state": self._current_patient["emotional_state"],
            "emotional_label": self._current_patient["emotional_label"],
            "location": self._current_patient["location"],
        }

        task_instructions = (
            f"You are a medical communicator. A patient named {self._current_patient['name']} "
            f"(age {self._current_patient['age']}, speaks {self._current_patient['language']}, "
            f"education: {self._current_patient['education_level']}, "
            f"emotional state: {self._current_patient['emotional_label']}) "
            f"has received a medical diagnosis.\n\n"
            f"STEP 1: Explain the following medical diagnosis to this patient "
            f"in {self._current_patient['language']}. "
            f"Use simple language appropriate for {self._current_patient['education_level']} education level. "
            f"The patient is feeling {self._current_patient['emotional_label'].lower()}, "
            f"so adjust your tone accordingly.\n\n"
            f"Provide your explanation in the 'explanation' field of your action."
        )

        return MedbridgeObservation(
            patient_profile=patient_profile,
            medical_report=self._current_report["diagnosis_technical"],
            diagnosis_name=self._current_report["diagnosis_name"],
            severity=self._current_report["severity"],
            task=task_instructions,
            current_step=0,
            episode_phase="awaiting_explanation",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: MedbridgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedbridgeObservation:
        """
        Execute one step in the environment.

        Step 1 (awaiting_explanation):
            Agent provides explanation → environment returns follow-up question.
            Partial rewards (accuracy, simplicity, tone, language) are computed.

        Step 2 (awaiting_followup):
            Agent provides follow-up answer → environment returns full reward breakdown.
            Follow-up reward is computed, total reward is calculated, episode ends.

        Returns:
            MedbridgeObservation with appropriate fields for the current step.
        """
        self._state.step_count += 1
        rewards_mod = _get_rewards()


        # ──────────────────────────────────────────────
        # TIMEOUT PROTECTION: Max 4 steps then force-end
        # ──────────────────────────────────────────────
        if self._state.step_count > 4:
            self._phase = "complete"
            return MedbridgeObservation(
                current_step=self._state.step_count,
                episode_phase="complete",
                done=True,
                reward=0.0,
                reward_breakdown={"accuracy": 0, "simplicity": 0, "tone": 0, "language": 0, "followup": 0, "total": 0},
                metadata={"error": "Episode timeout — maximum steps exceeded."},
            )

        # ──────────────────────────────────────────────
        # STEP 1: Receive explanation, return follow-up question
        # ──────────────────────────────────────────────
        if self._phase == "awaiting_explanation":
            self._explanation = action.explanation

            if not self._explanation or len(self._explanation.strip()) == 0:
                # Empty explanation — penalize and end
                self._phase = "complete"
                return MedbridgeObservation(
                    current_step=self._state.step_count,
                    episode_phase="complete",
                    done=True,
                    reward=0.0,
                    reward_breakdown={"accuracy": 0, "simplicity": 0, "tone": 0, "language": 0, "followup": 0, "total": 0},
                    metadata={"error": "Empty explanation provided."},
                )

            # Compute partial rewards for the explanation (4 of 5 dimensions)
            try:
                acc = rewards_mod.score_accuracy(self._explanation, self._current_report)
                simp = rewards_mod.score_simplicity(self._explanation, self._current_patient)
                tone = rewards_mod.score_tone(self._explanation, self._current_patient)
                lang = rewards_mod.score_language(self._explanation, self._current_patient)
            except Exception as e:
                acc, simp, tone, lang = 0.0, 0.5, 0.5, 0.0

            self._explanation_rewards = {
                "accuracy": round(acc, 2),
                "simplicity": round(simp, 2),
                "tone": round(tone, 2),
                "language": round(lang, 2),
            }

            # Move to step 2
            self._phase = "awaiting_followup"

            # Build follow-up question observation
            followup_question = "Do you have any other questions?"
            if self._followup_data:
                followup_question = self._followup_data["question"]

            task_step2 = (
                f"The patient has a follow-up question. "
                f"Answer it honestly, empathetically, and in {self._current_patient['language']}.\n\n"
                f"Patient's question: {followup_question}\n\n"
                f"Provide your answer in the 'followup_answer' field of your action."
            )

            # Return intermediate observation — partial reward as signal
            partial_reward = (acc * 0.30) + (simp * 0.20) + (tone * 0.20) + (lang * 0.20)

            return MedbridgeObservation(
                followup_question=followup_question,
                task=task_step2,
                current_step=1,
                episode_phase="awaiting_followup",
                done=False,
                reward=round(partial_reward, 4),
                reward_breakdown=self._explanation_rewards,
                metadata={
                    "info": "Step 1 complete. Now answer the follow-up question.",
                    "partial_scores": self._explanation_rewards,
                },
            )

        # ──────────────────────────────────────────────
        # STEP 2: Receive follow-up answer, compute final reward
        # ──────────────────────────────────────────────
        elif self._phase == "awaiting_followup":
            self._followup_answer = action.followup_answer

            if not self._followup_answer or len(self._followup_answer.strip()) == 0:
                self._followup_answer = ""

            # Compute follow-up reward
            try:
                foll = rewards_mod.score_followup(
                    self._current_report["id"],
                    self._followup_answer
                )
            except Exception:
                foll = 0.0

            # Assemble complete reward breakdown
            acc = self._explanation_rewards.get("accuracy", 0.0)
            simp = self._explanation_rewards.get("simplicity", 0.0)
            tone = self._explanation_rewards.get("tone", 0.0)
            lang = self._explanation_rewards.get("language", 0.0)

            # Anti-hacking: if accuracy is 0, total reward is 0
            if acc == 0.0:
                total = 0.0
            else:
                total = (acc * 0.30) + (simp * 0.20) + (tone * 0.20) + (lang * 0.20) + (foll * 0.10)

            self._reward_breakdown = {
                "accuracy": round(acc, 2),
                "simplicity": round(simp, 2),
                "tone": round(tone, 2),
                "language": round(lang, 2),
                "followup": round(foll, 2),
                "total": round(total, 4),
            }

            self._phase = "complete"

            return MedbridgeObservation(
                current_step=2,
                episode_phase="complete",
                done=True,
                reward=round(total, 4),
                reward_breakdown=self._reward_breakdown,
                metadata={
                    "explanation_length": len(self._explanation),
                    "followup_answer_length": len(self._followup_answer),
                    "patient_language": self._current_patient["language"],
                    "diagnosis": self._current_report["diagnosis_name"],
                    "severity": self._current_report["severity"],
                },
            )

        # ──────────────────────────────────────────────
        # Episode already complete — no more actions
        # ──────────────────────────────────────────────
        else:
            return MedbridgeObservation(
                current_step=self._state.step_count,
                episode_phase="complete",
                done=True,
                reward=0.0,
                metadata={"error": "Episode already complete. Call reset() to start a new episode."},
            )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_metadata(self):
        """Return metadata about this environment for the OpenEnv UI."""
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="MedBridge",
            description=(
                "A medical communication RL environment that trains LLMs to explain "
                "complex diagnoses to patients in their native Indian language, "
                "at their reading level, matching their emotional state."
            ),
            version="1.0.0",
        )
