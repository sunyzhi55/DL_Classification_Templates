import argparse
import torch
import sys
sys.path.append('../')
from pathlib import Path
from models.get_model import get_model

def compute_params_flops(model, input_res=(3, 224, 224), device='cpu'):
    """Compute Params and FLOPs using ptflops if available.

    Returns a dict with keys: 'macs', 'params', 'macs_str', 'params_str'
    """
    try:
        from ptflops import get_model_complexity_info
    except Exception as e:
        raise RuntimeError("ptflops is required for computing FLOPs. Install with: pip install ptflops") from e

    # Move model to device for consistent behavior
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
    try:
        model.cpu()
        macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False, verbose=False)
        macs_str, params_str = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
    finally:
        # try to restore original device
        try:
            model.to(model_device)
        except Exception:
            pass

    # ptflops returns MACs. Many people report FLOPs = 2 * MACs for multiply-adds.
    flops = float(macs) * 2.0

    return {
        'macs': float(macs),
        'flops': flops,
        'params': float(params),
        'macs_str': macs_str,
        'params_str': params_str,
    }


def _load_model_via_get_model(num_classes, checkpoint_path, device_str):
    """Import project get_model and build model instance used by this repo."""
    # Import here to avoid top-level import issues
    from models.get_model import get_model
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes, checkpoint_path, device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Compute model Params and FLOPs for models in this repo")
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output', type=str, default=None, help='Optional path to save result as txt')

    args = parser.parse_args()

    model = _load_model_via_get_model(args.num_classes, args.checkpoint, args.device)

    stats = compute_params_flops(model, input_res=(3, args.img_size, args.img_size), device=args.device)

    lines = []
    lines.append(f"Params (raw): {stats['params']}")
    lines.append(f"Params (str): {stats['params_str']}")
    lines.append(f"MACs (raw): {stats['macs']}")
    lines.append(f"MACs (str): {stats['macs_str']}")
    lines.append(f"Estimated FLOPs (2*MACs): {stats['flops']}")

    out = "\n".join(lines)
    print(out)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding='utf-8')
        print(f"Saved stats to {out_path}")


if __name__ == '__main__':
    main()
"""
Resnet34 on Oxford Flowers102
Command:
Params (raw): 21336998.0
Params (str): 21.34 M
MACs (raw): 3679558758.0
MACs (str): 3.68 GMac
Estimated FLOPs (2*MACs): 7359117516.0
"""