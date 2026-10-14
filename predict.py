import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from PIL import Image, ImageOps
from core import load_checkpoint, transform


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default=str(Path(__file__).resolve().parent / "weights/facecheck-ensemble.pt")); p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--name-column",default="filename"); p.add_argument("--prob-column",default="fake_probability")
    p.add_argument("--probability-of",choices=["fake","real"],default="fake")
    args=p.parse_args()
    torch.set_num_threads(8)
    if args.batch_size < 1: raise ValueError("Batch size must be positive")
    if args.name_column==args.prob_column: raise ValueError("Column names must differ")
    model,ck=load_checkpoint(args.checkpoint,args.device)
    tf=transform(ck["config"]["image_size"])
    files=sorted(x for x in args.input.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png"})
    if not files: raise ValueError("No images")
    names=[x.name for x in files]
    if len(set(names))!=len(names): raise ValueError("Duplicate basename: confirm official naming schema")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f,torch.inference_mode():
        w=csv.writer(f); w.writerow([args.name_column,args.prob_column])
        for start in range(0,len(files),args.batch_size):
            batch=[]
            for path in files[start:start+args.batch_size]:
                try:
                    with Image.open(path) as im: batch.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
                except Exception as e: raise RuntimeError(f"Unreadable required image: {path.name}") from e
            probs=model(torch.stack(batch).to(args.device)).flatten().sigmoid().cpu()
            if not torch.isfinite(probs).all(): raise ValueError("Non-finite probabilities")
            if args.probability_of=="real": probs=1-probs
            w.writerows(zip(names[start:start+len(batch)],probs.tolist()))
    tmp.replace(args.output)
    print(f"Predicted {len(files)} images, probability of {args.probability_of}, output={args.output}")


if __name__=="__main__": main()
