"""Download pinned official REVE assets for local source review; never execute them."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from huggingface_hub import HfApi,hf_hub_download
from gruxnet.data import sha256,write_json

REVISIONS = {"brain-bzh/reve-base":"dc2a075c287bb2f6c04ee5875bd79535a0f7dba6",
             "brain-bzh/reve-positions":"befa5b57a455b77cf302daf610c2e9ed8140bace"}


def download(output,weights=False):
    output.mkdir(parents=True,exist_ok=True)
    api = HfApi(token=False)
    records = []
    for repo,revision in REVISIONS.items():
        info = api.model_info(repo,revision=revision,files_metadata=True)
        if info.sha!=revision:
            raise ValueError("Changed pinned revision")
        for item in info.siblings:
            if item.rfilename==".gitattributes" or (item.rfilename=="model.safetensors" and not weights):
                continue
            path = Path(hf_hub_download(repo,item.rfilename,revision=revision,token=False,local_dir=output/repo.split('/')[1]))
            actual = sha256(path)
            if item.size!=path.stat().st_size or (item.lfs and actual!=item.lfs.sha256):
                raise ValueError("Changed official download")
            records.append({"repository":repo,"revision":revision,"file":item.rfilename,
                            "local_file":path.relative_to(output).as_posix(),"size_bytes":item.size,"sha256":actual,
                            "expected_lfs_sha256":item.lfs.sha256 if item.lfs else None})
            print(f"Downloaded pinned {repo}/{item.rfilename} ({item.size} bytes)",flush=True)
    dataset = api.dataset_info("brain-bzh/reve-dataset",files_metadata=True)
    readme = Path(hf_hub_download("brain-bzh/reve-dataset","README.md",repo_type="dataset",revision=dataset.sha,token=False,local_dir=output/"reve-dataset"))
    metadata = {"development_only":True,"weights_downloaded":weights,"model_revisions":REVISIONS,"files":records,
                "pretraining_card":{"repository":"brain-bzh/reve-dataset","revision":dataset.sha,"file":"README.md","sha256":sha256(readme)},
                "execution":"None; source review and separate local-only inference are required","target_overlap":"Not established by downloading assets"}
    write_json(output/"download_manifest.json",metadata)
    return {"downloaded_files":len(records),"weights_downloaded":weights,"manifest":str(output/"download_manifest.json")}


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",action="store_true")
    args=parser.parse_args()
    print(json.dumps(download(args.output.resolve(),args.weights)))
