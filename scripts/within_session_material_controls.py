from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet import material_controls as control
from gruxnet.data import write_json

if __name__=="__main__":
    parser=ArgumentParser(description="Matched within-session participant/material exposure controls")
    parser.add_argument("action",choices=["plan","run","verify","analyze"])
    for name in ("cache","reve-run","output","plan"): parser.add_argument("--"+name,type=Path)
    parser.add_argument("--device",default="cuda"); a=parser.parse_args()
    if a.action=="plan":
        if a.plan.exists(): raise FileExistsError("Existing material declaration")
        write_json(a.plan,control.plan()); result={"saved":str(a.plan)}
    elif a.action=="run":
        r=control.run(a.cache,a.reve_run,a.output,a.plan,a.device); result={"completed":True,"models":len(r["models"]),"contrasts":len(r["contrasts"])}
    elif a.action=="verify": result=control.verify(a.cache,a.reve_run,a.output,a.device)
    else:
        r=control.analyze(a.output); result={"models":len(r["models"]),"contrasts":len(r["contrasts"])}
    print(json.dumps(result,indent=2))
