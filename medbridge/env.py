# env.py — Main MedBridge RL environment class.
# Implements reset(), step(), state(), and render() methods.
# Generates patient scenarios and scores agent explanations using all 5 reward functions.

import random
import sys

sys.path.append(".")
from patients import generate_patient
from reports import get_random_report
from rewards import compute_total_reward
from followups import get_followup


class MedBridgeEnv:
    """
    MedBridge Reinforcement Learning Environment.

    Each episode:
      1. reset() generates a new patient + medical report scenario.
      2. The agent receives the observation (patient profile + diagnosis).
      3. The agent calls step() with an explanation and follow-up answer.
      4. The environment scores the response across 5 reward dimensions.
      5. The episode ends after 1 step (or times out after 3 steps).
    """

    def __init__(self):
        """Initialize the environment with empty state."""
        self.current_patient = None
        self.current_report = None
        self.conversation_history = []
        self.step_count = 0
        self.episode_done = False
        self.last_observation = None
        self.last_reward_breakdown = None

    def reset(self):
        """
        Reset the environment and generate a fresh patient scenario.

        Returns:
            observation: dictionary containing patient profile, medical report,
                         and the task description for the agent.
        """
        # Generate a new random patient and medical report
        self.current_patient = generate_patient()
        self.current_report = get_random_report()

        # Reset episode state
        self.conversation_history = []
        self.step_count = 0
        self.episode_done = False
        self.last_reward_breakdown = None

        # Build the observation that the agent will see
        observation = {
            "patient_profile": {
                "name": self.current_patient["name"],
                "age": self.current_patient["age"],
                "language": self.current_patient["language"],
                "education_level": self.current_patient["education_level"],
                "emotional_state": self.current_patient["emotional_state"],
                "emotional_label": self.current_patient["emotional_label"],
                "location": self.current_patient["location"]
            },
            "medical_report": self.current_report["diagnosis_technical"],
            "diagnosis_name": self.current_report["diagnosis_name"],
            "severity": self.current_report["severity"],
            "task": (
                "Explain this medical diagnosis to the patient in their native language, "
                "at their education level, matching their emotional state. "
                "Then answer their follow-up question."
            )
        }

        self.last_observation = observation
        return observation

    def step(self, action):
        """
        Execute one step: receive the agent's explanation and score it.

        Args:
            action: dictionary with keys:
                - "explanation": the agent's medical explanation string
                - "followup_answer": the agent's answer to the patient's follow-up question

        Returns:
            tuple of (observation, reward, done, info):
                - observation: dict with follow-up question and conversation history
                - reward: float total reward (0.0 to 1.0)
                - done: boolean indicating if episode is finished
                - info: dict with individual reward breakdown
        """
        self.step_count += 1

        # Validate that action is a dictionary
        if not isinstance(action, dict):
            self.episode_done = True
            error_info = {"error": "Action must be a dictionary with 'explanation' and 'followup_answer' keys."}
            return (error_info, 0.0, True, error_info)

        # Validate that explanation key exists
        if "explanation" not in action:
            self.episode_done = True
            error_info = {"error": "Action missing required key: 'explanation'."}
            return (error_info, 0.0, True, error_info)

        # Validate that followup_answer key exists
        if "followup_answer" not in action:
            self.episode_done = True
            error_info = {"error": "Action missing required key: 'followup_answer'."}
            return (error_info, 0.0, True, error_info)

        # Compute all 5 reward scores
        try:
            rewards = compute_total_reward(
                action["explanation"],
                action["followup_answer"],
                self.current_patient,
                self.current_report
            )
        except Exception as e:
            self.episode_done = True
            error_info = {"error": f"Reward computation failed: {str(e)}"}
            return (error_info, 0.0, True, error_info)

        self.last_reward_breakdown = rewards

        # Get the follow-up question for this report in the patient's language
        followup_data = get_followup(
            self.current_report["id"],
            self.current_patient["language_code"]
        )

        # If no followup found, use a fallback
        if followup_data is None:
            followup_data = {
                "question": "Do you have any other questions?",
                "must_include": [],
                "must_not_include": [],
                "scoring_notes": "Fallback question — no specific scoring."
            }

        # Record the conversation turn
        self.conversation_history.append({
            "agent_explanation": action["explanation"],
            "patient_followup_question": followup_data["question"],
            "agent_followup_answer": action["followup_answer"]
        })

        # Check for timeout — episodes end after 1 step normally, max 3 steps allowed
        if self.step_count > 3:
            self.episode_done = True
            rewards["total"] = 0.0  # Timeout penalty
        else:
            self.episode_done = True  # Normal single-step episode

        # Build the step observation
        observation = {
            "followup_question": followup_data["question"],
            "conversation_history": self.conversation_history,
            "reward_breakdown": rewards
        }

        return (observation, rewards["total"], self.episode_done, rewards)

    def state(self):
        """
        Return the current internal state of the environment.

        Returns:
            dictionary with patient info, report info, step count,
            conversation history, and reward breakdown.
        """
        # Build a safe report summary (avoid exposing full key_facts to agent)
        report_summary = None
        if self.current_report is not None:
            report_summary = {
                "id": self.current_report["id"],
                "diagnosis_name": self.current_report["diagnosis_name"],
                "severity": self.current_report["severity"]
            }

        return {
            "current_patient": self.current_patient,
            "current_report": report_summary,
            "step_count": self.step_count,
            "episode_done": self.episode_done,
            "conversation_history": self.conversation_history,
            "last_reward_breakdown": self.last_reward_breakdown
        }

    def render(self):
        """
        Print a human-readable summary of the current environment state.
        Useful for debugging and demonstration.
        """
        print("-" * 50)
        print("MEDBRIDGE ENVIRONMENT STATE")
        print("-" * 50)

        # Show patient profile
        if self.current_patient is not None:
            p = self.current_patient
            print(f"Patient: {p['name']}, Age {p['age']}, {p['gender']}")
            print(f"Language: {p['language']} ({p['language_code']})")
            print(f"Education: {p['education_level']} (target grade: {p['target_grade']})")
            print(f"Emotional State: {p['emotional_state']}/10 ({p['emotional_label']})")
            print(f"Location: {p['location']}")
        else:
            print("No patient loaded. Call reset() first.")

        print("")

        # Show report info
        if self.current_report is not None:
            r = self.current_report
            print(f"Diagnosis: {r['diagnosis_name']}")
            print(f"Severity: {r['severity']}")
            print(f"Category: {r['category']}")
        else:
            print("No report loaded.")

        print("")

        # Show conversation history
        if len(self.conversation_history) > 0:
            print(f"Conversation ({len(self.conversation_history)} turn(s)):")
            for i, turn in enumerate(self.conversation_history):
                print(f"  Turn {i + 1}:")
                print(f"    Agent explanation: {turn['agent_explanation'][:80]}...")
                print(f"    Patient question:  {turn['patient_followup_question']}")
                print(f"    Agent answer:      {turn['agent_followup_answer'][:80]}...")
        else:
            print("No conversation yet.")

        print("")

        # Show reward breakdown
        if self.last_reward_breakdown is not None:
            print("Reward Breakdown:")
            for key, value in self.last_reward_breakdown.items():
                print(f"  {key}: {value:.2f}")
        else:
            print("No rewards computed yet.")

        print("-" * 50)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("Testing MedBridgeEnv")
    print("=" * 60)

    # Create environment
    env = MedBridgeEnv()

    # Test reset
    print("\nTest 1: Reset environment")
    obs = env.reset()
    print("Patient:", obs["patient_profile"]["name"],
          obs["patient_profile"]["age"],
          obs["patient_profile"]["language"])
    print("Diagnosis:", obs["diagnosis_name"])
    print("Severity:", obs["severity"])
    print("Task:", obs["task"][:50], "...")

    # Test step with fake action
    print("\nTest 2: Step with fake action")
    fake_action = {
        "explanation": (
            f"உங்களுக்கு {obs['diagnosis_name']} உள்ளது. "
            "இது கவனமாக சிகிச்சை அளிக்கப்பட வேண்டும். "
            "மருந்து ஒவ்வொரு நாளும் எடுக்க வேண்டும்."
        ),
        "followup_answer": (
            "ஆம், நீங்கள் கவனமாக சிகிச்சையைப் பின்பற்றினால் "
            "முழுமையாக குணமடையலாம்."
        )
    }

    obs, reward, done, info = env.step(fake_action)
    print(f"Reward: {reward:.2f}")
    print(f"Episode done: {done}")
    print("Reward breakdown:")
    for key, value in info.items():
        print(f"  {key}: {value:.2f}")

    # Test state
    print("\nTest 3: Get current state")
    state = env.state()
    print("Steps taken:", state["step_count"])
    print("Episode done:", state["episode_done"])

    # Test render
    print("\nTest 4: Render environment")
    env.render()

    print("\n" + "=" * 60)
    print("env.py working correctly")
