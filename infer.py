import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from models.get_model import get_model
from utils.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, nargs='+', help='Path to image(s)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    args = parse_args()
    
    # 设置随机种子（推理时通常不需要，但为了完整性）
    set_global_seed(args.seed, deterministic=False)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = get_model(args.num_classes, args.checkpoint, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    transform = build_transform(args.img_size)

    for img_path in args.image:
        p = Path(img_path)
        if not p.exists():
            print(f"Image not found: {img_path}")
            continue
        img = Image.open(str(p)).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            topk_idx = prob.argsort()[-args.topk:][::-1]
            print(f"\nResults for {img_path}:")
            for idx in topk_idx:
                print(f"  class {idx}: {prob[idx]:.4f}")
