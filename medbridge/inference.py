import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


class MedbridgeInference:
    """
    Loads Qwen2.5-3B-Instruct + LoRA adapter.
    Supports multi-turn conversation where the model acts as the DOCTOR.
    """

    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_path="model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MedBridge] Loading base model: {self.BASE_MODEL} on {device}...")

        tokenizer_path = self.model_path if os.path.exists(
            os.path.join(self.model_path, "tokenizer_config.json")
        ) else self.BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        adapter_path = self.model_path
        if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
            print(f"[MedBridge] Applying LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("[MedBridge] LoRA adapter loaded.")
        else:
            print(f"[MedBridge] WARNING: No adapter at {adapter_path}. Using base model.")

        if device == "cpu":
            self.model = self.model.float()

        self.model.eval()
        self.is_loaded = True
        print("[MedBridge] Model ready for inference.")

    def chat(self, messages):
        """
        Generate a doctor response given a list of chat messages.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        The model acts as the assistant (doctor).
        """
        if not self.is_loaded:
            self.load()

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()
