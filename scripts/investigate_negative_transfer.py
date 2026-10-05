"""Run the predeclared feature-level neural/source/exposure controls."""
from argparse import ArgumentParser
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.transfer_controls import investigate

if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for name in ["pack","common-cache","plan","output"]:
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    investigate(args.pack,args.common_cache,args.plan,args.output)
