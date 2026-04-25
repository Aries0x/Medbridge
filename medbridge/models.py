# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the MedBridge RL Environment.

MedBridge is a 2-step medical communication environment:
  Step 1: Agent receives patient profile + medical report → sends explanation
  Step 2: Agent receives follow-up question → sends follow-up answer
"""

from typing import Dict, List, Optional, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MedbridgeAction(Action):
    """
    Action sent by the agent at each step.

    Step 1: Agent provides 'explanation' (medical diagnosis explanation for patient).
    Step 2: Agent provides 'followup_answer' (answer to patient's follow-up question).

    The agent should fill the relevant field for the current step.
    """

    explanation: str = Field(
        default="",
        description="Agent's explanation of the medical diagnosis to the patient (used in Step 1)"
    )
    followup_answer: str = Field(
        default="",
        description="Agent's answer to the patient's follow-up question (used in Step 2)"
    )


class MedbridgeObservation(Observation):
    """
    Observation returned to the agent at each step.

    On reset (Step 0): Contains patient_profile, medical_report, task instructions.
    After Step 1: Contains followup_question for the agent to answer.
    After Step 2: Contains full reward_breakdown with episode complete.
    """

    # --- Present on reset ---
    patient_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient demographics: name, age, language, education, emotional_state, location"
    )
    medical_report: str = Field(
        default="",
        description="The technical medical diagnosis text that must be explained"
    )
    diagnosis_name: str = Field(
        default="",
        description="Simple readable name of the diagnosis"
    )
    severity: str = Field(
        default="",
        description="Severity level: Mild, Moderate, Serious, or Critical"
    )
    task: str = Field(
        default="",
        description="Task instructions for the agent"
    )

    # --- Present after Step 1 ---
    followup_question: str = Field(
        default="",
        description="The patient's follow-up question (appears after Step 1)"
    )

    # --- Present after Step 2 ---
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual reward scores: accuracy, simplicity, tone, language, followup, total"
    )

    # --- Step tracking ---
    current_step: int = Field(
        default=0,
        description="Current step in the episode (0=reset, 1=after explanation, 2=after followup)"
    )
    episode_phase: str = Field(
        default="reset",
        description="Current phase: 'reset', 'awaiting_explanation', 'awaiting_followup', 'complete'"
    )
