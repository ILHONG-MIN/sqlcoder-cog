from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor(BasePredictor):
    def setup(self):
        # 모델 로드 (예: defog/sqlcoder-7b-2)
        self.tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "defog/sqlcoder-7b-2",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def predict(
        self,
        prompt: str = Input(description="Full prompt for SQL generation"),
        max_new_tokens: int = Input(default=512, ge=1, le=2048),
        temperature: float = Input(default=0.0),
        top_p: float = Input(default=1.0),
        top_k: int = Input(default=50),
    ) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True
        )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result
