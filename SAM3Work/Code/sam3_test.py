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

    draw = ImageDraw.Draw(image)
    for box in boxes.float().cpu().tolist():
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
    image.show()