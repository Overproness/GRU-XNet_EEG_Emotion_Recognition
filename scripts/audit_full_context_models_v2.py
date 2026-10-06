"""Audit corrected local reference in evaluation AND training mode before fits."""
import importlib.util
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.full_context_models_v2 import FullControl, copy_legacy
from gruxnet.data import sha256, write_json
from gruxnet.train import seed_everything

REPO=Path(__file__).resolve().parents[1]

def audit():
    seed_everything(42); records=[]
    for name,relative,kind in (('gru','OurApproach/CBSAtt/model.py','gru_xnetDynamic'),('cbsatt_local','CBSAtt_og/model.py','CBSAtt')):
        path=REPO.parent/relative; spec=importlib.util.spec_from_file_location('audited_local',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for classes in (2,3):
            legacy=getattr(module,kind)(n_channels=14,n_freq_bins=37,n_time_bins=79,n_classes=classes).eval()
            model=FullControl(name,classes).eval(); copy_legacy(model,legacy); x=torch.randn(2,14,37,79)
            with torch.no_grad():
                a=legacy(x); a=a[0] if isinstance(a,tuple) else a; b=model(x)
            error=float((a-b).abs().max())
            if error>1e-5: raise ValueError('Historical forward mismatch')
            captured=[]; handle=model.cnn.drop.register_forward_pre_hook(lambda _,args: captured.append(tuple(args[0].shape)))
            model.train()
            with torch.no_grad(): model(x)
            handle.remove(); expected=(2,14*128,1,1) if name=='cbsatt_local' else (2,14*128,4,9)
            if captured!=[expected]: raise ValueError('Dropout pooling order differs from original')
            records.append({'model':name,'classes':classes,'local_source':relative,'local_source_sha256':sha256(path),
                            'maximum_eval_logit_error':error,'training_dropout_input_shape':captured[0],
                            'scope':'Weight-mapped eval forward; training dropout placement checked. Distinct channel parameters/statistics retained; random draw ordering differs under vectorization, no bitwise optimization reproduction claim.'})
            del legacy,model,x
    result={'passed':True,'revision':2,'source_sha256':sha256(Path(__file__)),'model_source_sha256':sha256(REPO/'gruxnet/full_context_models_v2.py'),'records':records}
    write_json(REPO.parent/'publication_runs/full_context_v2_model_audit.json',result)
    print(result)

if __name__=='__main__': audit()
