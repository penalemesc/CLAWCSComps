import torch

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="sam3.pt",
    bpe_path="/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    load_from_HF=False,                                 
    eval_mode=True                    
)
processor = Sam3Processor(model)

image = Image.open("lambo.jpg")
with torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="Wheels")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    
if __name__ == "__main__":
    print("Masks:", masks)
    print("Boxes:", boxes)
    print("Scores:", scores)