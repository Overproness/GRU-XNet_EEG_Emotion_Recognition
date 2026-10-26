"""Pin and inspect author source as text/AST only; never import or execute it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PACKET=REPO/'results/development/faced_eegnet_preflight_2026-10-11'
REVISION='156d1efc335c7f399ac7e1647f0245d17afd2486'


def audit(author_directory):
    receipts=json.loads((PACKET/'closest_method_source_receipts.json').read_bytes())
    sources={};trees={}
    for entry in receipts['downloads']['files']:
        path=Path(author_directory)/entry['path'];content=path.read_bytes()
        if len(content)!=entry['bytes'] or hashlib.sha256(content).hexdigest()!=entry['sha256']:
            raise ValueError('Pinned author source changed: '+entry['path'])
        sources[entry['path']]=content.decode('utf-8')
        if path.suffix=='.py':trees[entry['path']]=ast.parse(sources[entry['path']])
    if receipts['downloads']['revision']!=REVISION:raise ValueError('Different author revision')
    def function(path,name):
        nodes=[n for n in ast.walk(trees[path]) if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:raise ValueError('Ambiguous source function')
        n=nodes[0];return n,ast.get_source_segment(sources[path],n)
    labels='decoding/labels.py';classic='decoding/classical/classification.py'
    paths=[(labels,'load_subjective_labels'),(classic,'_run_fold_cv'),(classic,'_cross_video_folds'),(classic,'_run_matched_split'),(classic,'run_cross_video_permutation'),(classic,'run_single_video'),('decoding/classical/normalization.py','run_normalization'),('decoding/clisa/classify.py','_run_classify')]
    evidence={}
    for path,name in paths:
        node,text=function(path,name)
        evidence[name]={'path':path,'first_line':node.lineno,'last_line':node.end_lineno,
                        'url':f'https://github.com/moritz-gerster/faced-stimulus-confound/blob/{REVISION}/{path}#L{node.lineno}'}
    _,text=function(labels,'load_subjective_labels')
    if '[:8]' not in text or 'np.argmax(scores)' not in text:raise ValueError('Individual categorical target changed')
    _,text=function(classic,'_run_matched_split')
    if not all(pattern in text for pattern in ('_zscore_per_subject(data)','data[:, train_idx, :]','data[:, test_idx, :]','inner_splits','de_lds_fold0.mat')):
        raise ValueError('Cross-video method changed')
    _,text=function(classic,'_cross_video_folds')
    if not all(pattern in text for pattern in ('test_vids = [g[k] for g in groups]','inner_splits','N_MATCHED_FOLDS')):
        raise ValueError('Author fold construction changed')
    # Independently reconstruct only metadata clip folds from literal class counts.
    groups=[list(range(a,a+3)) for a in (1,4,7,10,13,17,20,23,26)]
    catalogue={r['clip']:r for r in json.loads((REPO/'results/development/faced_backup_2026-10-10/stimulus_metadata.json').read_bytes())['records']}
    folds=[]
    for k in range(3):
        test=[g[k] for g in groups];train=[v for g in groups for i,v in enumerate(g) if i!=k]
        trainfamilies={catalogue[v]['source_film_sha256'] for v in train}
        overlap=[v for v in test if catalogue[v]['source_film_sha256'] in trainfamilies]
        folds.append({'fold':k,'train_clips':train,'test_clips':test,'same_catalogue_title_test_clips':overlap,
                      'same_participant_axis_retained':True,'nested_C_inner_clip_folds':2})
    if [f['same_catalogue_title_test_clips'] for f in folds]!=[[7,13],[8,14],[15]]:
        raise ValueError('Catalogue-title overlap differs from inspected source')
    return {'revision':REVISION,'commit_utc':'2026-08-10T08:48:41Z','source_files_hash_verified':len(sources),
            'source_bytes_verified':sum(e['bytes'] for e in receipts['downloads']['files']),
            'source_code_executed':False,'author_figures_or_numeric_outcomes_opened':False,
            'individual_target':'argmax of first eight categorical self-report scores, not continuous item-10 valence',
            'source_evidence':evidence,'cross_video_metadata_folds':folds,
            'cross_video_and_nested_C_already_present':True,
            'our_operational_differences':['continuous item-10 retrospective valence','new participants together with catalogue-title-held families','available-metadata context comparison and acquisition fallback','source-only transformations and preserved confirmation'],
            'distinction_establishes_novel_contribution':False,
            'published_v1_to_august_code_equivalence':'unverified; publisher full text/PDF remains inaccessible',
            'published_score_reproduction':'not attempted or claimed'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--author-directory',type=Path,required=True)
    args=parser.parse_args();result=audit(args.author_directory)
    (PACKET/'closest_method_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({'source_files_verified':result['source_files_hash_verified'],'cross_video_title_overlap':[f['same_catalogue_title_test_clips'] for f in result['cross_video_metadata_folds']],'author_code_executed':False})
