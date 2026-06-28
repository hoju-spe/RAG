from openai import OpenAI


def analyze_tone_with_openai(client: OpenAI, model: str, answer: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
당신은 면접관 관점에서 지원자의 답변 톤을 분석하는 코치입니다.
아래 형식으로 간결하게 답하세요.

tone: 답변의 감정 톤
confidence: 자신감 수준
positive: 긍정적인 점
negative: 아쉬운 점
summary: 총평
""",
            },
            {"role": "user", "content": f"답변: {answer}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


class QwenToneAnalyzer:
    """Optional GPU-based local tone analyzer.

    This class is intentionally not used by default. Install optional dependencies
    and call it only in a CUDA runtime such as Colab T4.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": "cuda"},
        )

    def analyze(self, answer: str) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": "당신은 냉철한 면접관입니다. tone, confidence, positive, negative, summary 형식으로 답하세요.",
            },
            {"role": "user", "content": f"답변: {answer}"},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.1,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
