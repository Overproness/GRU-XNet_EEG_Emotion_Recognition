from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet import reve_probe as probe
from gruxnet.data import write_json

if __name__=="__main__":
    parser=ArgumentParser(description="Audited local-only frozen pretrained/random REVE controls")
    parser.add_argument("action",choices=["audit","prepare","pilot","plan","extract","run","analyze","verify"])
    for name in ("assets","cache","data-root","temporal-cache","output","plan"):
        parser.add_argument("--"+name,type=Path)
    a=parser.parse_args()
    if a.action=="audit": result=probe.audit_assets(a.assets.resolve())
    elif a.action=="prepare": result=probe.prepare(a.data_root,a.temporal_cache,a.cache)
    elif a.action=="pilot": result=probe.pilot(a.assets.resolve(),a.cache)
    elif a.action=="plan":
        if a.plan.exists(): raise FileExistsError("Existing probe declaration")
        write_json(a.plan,probe.plan()); result={"saved":str(a.plan)}
    elif a.action=="extract": result=probe.extract(a.assets.resolve(),a.cache,a.output,a.plan)
    elif a.action=="run": result=probe.run(a.output)
    elif a.action=="analyze": result=probe.analyze(a.output)
    else:
        from scripts.verify_reve_frozen_probe import verify
        result=verify(a.assets.resolve(),a.cache,a.output)
    print(json.dumps(result,indent=2))
