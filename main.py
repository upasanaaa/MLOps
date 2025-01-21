from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
    yield
    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs

app = FastAPI(lifespan=lifespan)

@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    try:
        image = Image.open(data.file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return {"captions": [caption.strip() for caption in captions]}
    except Exception as e:
        return {"error": str(e)}
