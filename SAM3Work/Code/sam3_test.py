import os
import torch
from PIL import ImageDraw
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("Building model...")

# This takes about 4-5 minutes
model = build_sam3_image_model(
    checkpoint_path="sam3.pt",
    bpe_path="/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    load_from_HF=False,                                 
    eval_mode=True                    
)
processor = Sam3Processor(model)

print("Model built successfully.")

while True:
    image_name = input("Image Name: ")
    image = Image.open(image_name)
    
    prompt = input("Prompt: ")
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print("Masks:", masks)
    print("Boxes:", boxes)
    print("Scores:", scores)

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for box, score in zip(boxes.float().cpu().tolist(), scores.float().cpu().tolist()):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=4)
        draw.text((x0 + 4, y0 + 4), f"{score:.2f}", fill="lime")

    out_path = f"{os.path.splitext(os.path.basename(image_name))[0]}_annotated.png"
    annotated.save(out_path)
    print(f"Saved annotated image to {os.path.abspath(out_path)}")
