"""Pin a small read-only author-code snapshot for protocol review; never import it."""
from datetime import datetime,timezone
from hashlib import sha256,sha1
import json
from pathlib import Path
from urllib.request import Request,urlopen

WORKSPACE=Path(__file__).resolve().parents[2]
OUTPUT=WORKSPACE/"publication_runs/material_prior_source_audit_2026-10-06"
FILES=("README.md","main-DEAP.py","cross_validation.py","trainer.py","prepare_data_DEAP.py")


def get(url):
    with urlopen(Request(url,headers={"User-Agent":"GRU-XNet-publication-protocol-review"}),timeout=30) as response:
        return response.read()


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/"manifest.json"
    if target.exists(): raise FileExistsError("Pinned source review already exists")
    repo="JZH98/Sera-code"
    commit=json.loads(get(f"https://api.github.com/repos/{repo}/commits?per_page=1"))[0]["sha"]
    records=[]
    for name in FILES:
        data=get(f"https://raw.githubusercontent.com/{repo}/{commit}/{name}")
        destination=OUTPUT/name
        if destination.exists() and destination.read_bytes()!=data: raise ValueError("Changed prior review file")
        destination.write_bytes(data)
        records.append({"name":name,"bytes":len(data),"sha256":sha256(data).hexdigest(),
                        "git_blob_sha1":sha1(f"blob {len(data)}\0".encode()+data).hexdigest(),
                        "source_url":f"https://github.com/{repo}/blob/{commit}/{name}"})
    result={"checked_utc":datetime.now(timezone.utc).isoformat(),"repo":repo,"commit":commit,
            "files":records,"snapshot_script_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"Read-only manual source/protocol inspection; author code not executed, imported or redistributed, no method reproduction"}
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": run()
