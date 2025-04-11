import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # 首先加载原始CLIP模型架构
checkpoint = torch.load("/home/face/kaichengyang/xiaoxinghu/Enhance-FineGrained/src/Outputs/negCLIP_my_new_data_B32_mse_distill_0.005/checkpoints/epoch_5.pt")
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
# 去掉state_dict中以'module.'开头的键
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)  # 加载训练好的权重
image = preprocess(Image.open("image.png")).unsqueeze(0).to(device)
text = clip.tokenize(["An athlete wearing yellow clothes is shooting a basketball.",
                       "An basketball wearing yellow clothes is shooting a athlete.",
                       "An athlete wearing red clothes is shooting a basketball.",
                        "Shooting a basketball is an athlete wearing yellow clothes."
                       ]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]