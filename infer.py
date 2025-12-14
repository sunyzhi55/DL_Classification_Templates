import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from models.get_model import get_model
from utils.reproducibility import set_global_seed
from data.dataset import get_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, nargs='+', help='Path to image(s)')
    group.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--checkpoint', type=str, 
                        default='/data3/wangchangmiao/shenxy/Code/DL_Classification_Templates/outputs_20251122_210542_oxfordFlowers/baseline_best_model_fold4.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_image(model, transform, img_path, device, args):
    p = Path(img_path)
    if not p.exists():
        print(f"Image not found: {img_path}")
        return
    img = Image.open(str(p)).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        topk_idx = prob.argsort()[-args.topk:][::-1]
        print(f"\nResults for {img_path}:")
        for idx in topk_idx:
            print(f"  class {idx}: {prob[idx]:.4f}")

if __name__ == '__main__':
    args = parse_args()
    
    set_global_seed(args.seed, deterministic=False)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(args.model_name, args.num_classes, args.checkpoint, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = build_transform(args.img_size)

    if args.image:
        for img_path in args.image:
            process_image(model, transform, img_path, device, args)
    elif args.folder:
        folder_path = Path(args.folder)
        for img_path in folder_path.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                process_image(model, transform, str(img_path), device, args)