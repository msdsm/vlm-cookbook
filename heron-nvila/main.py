from transformers import AutoConfig, AutoModel
from PIL import Image

model_path = "turing-motors/Heron-NVILA-Lite-15B"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")


image = Image.open("../assets/example.jpeg").convert("RGB")
response = model.generate_content([image, "Describe this image."])
print("="*10 + "response" + "="*10)
print(response)