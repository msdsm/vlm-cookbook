import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "sbintuitions/sarashina2-vision-8b"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype="auto",
    trust_remote_code=True,
)

message = [
    {
        "role": "user",
        "content": "この写真に写っているもので、最も有名と考えられる建築物は何でどこに写っていますか？"
    }
]

text_prompt = processor.apply_chat_template(message, add_generation_prompt=True)

image = Image.open("../assets/example.jpeg").convert("RGB")
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)
stopping_criteria = processor.get_stopping_criteria(["\n###"])

output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
    stopping_criteria=stopping_criteria,
)
generated_ids = [
    output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text[0])
