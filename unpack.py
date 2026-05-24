import torch
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Does not exist: {str(path)}")
    
    if path.suffix != '.pth':
        print(f"Incorrect suffix: {path.suffix}")
    
    ckpt = torch.load(path)
    model_ckpt = ckpt.get('model', ckpt)
    torch.save(model_ckpt, path.with_suffix('.pt'))
