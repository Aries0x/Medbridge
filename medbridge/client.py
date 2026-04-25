# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MedBridge Environment Client.

Connects to the MedBridge server via WebSocket for efficient
multi-step RL interactions (reset → explain → follow-up → done).
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MedbridgeAction, MedbridgeObservation


class MedbridgeEnv(
    EnvClient[MedbridgeAction, MedbridgeObservation, State]
):
    """
    Client for the MedBridge Environment.

    Connects to the deployed MedBridge FastAPI server via WebSocket.
    Each client gets its own dedicated environment session.

    Example:
        >>> with MedbridgeEnv(base_url="http://localhost:8000") as client:
        ...     # Step 0: Reset — get patient profile + medical report
        ...     result = client.reset()
        ...     print(result.observation.diagnosis_name)
        ...
        ...     # Step 1: Send explanation
        ...     result = client.step(MedbridgeAction(
        ...         explanation="Your blood sugar is high..."
        ...     ))
        ...     print(result.observation.followup_question)
        ...
        ...     # Step 2: Answer follow-up
        ...     result = client.step(MedbridgeAction(
        ...         followup_answer="Yes, you can still eat rice in small amounts."
        ...     ))
        ...     print(result.observation.reward_breakdown)
    """

    def _step_payload(self, action: MedbridgeAction) -> Dict:
        """
        Convert MedbridgeAction to JSON payload for the step message.

        Args:
            action: MedbridgeAction with explanation and/or followup_answer

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "explanation": action.explanation,
            "followup_answer": action.followup_answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MedbridgeObservation]:
        """
        Parse server response into StepResult[MedbridgeObservation].

        Handles both Step 1 (follow-up question) and Step 2 (reward breakdown)
        response formats.

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MedbridgeObservation
        """
        obs_data = payload.get("observation", {})

        observation = MedbridgeObservation(
            # Reset fields
            patient_profile=obs_data.get("patient_profile", {}),
            medical_report=obs_data.get("medical_report", ""),
            diagnosis_name=obs_data.get("diagnosis_name", ""),
            severity=obs_data.get("severity", ""),
            task=obs_data.get("task", ""),
            # Step 1 fields
            followup_question=obs_data.get("followup_question", ""),
            # Step 2 fields
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            # Tracking fields
            current_step=obs_data.get("current_step", 0),
            episode_phase=obs_data.get("episode_phase", ""),
            # Standard fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
