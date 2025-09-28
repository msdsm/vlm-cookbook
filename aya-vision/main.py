from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# need to run 'huggingface-cli login'
model_id = "CohereLabs/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "../assets/example.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=300, 
    do_sample=True, 
    temperature=0.3,
)

print("="*10 + "response" + "="*10)
print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
