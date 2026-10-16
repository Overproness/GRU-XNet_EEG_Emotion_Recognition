"""Export figures, public supporting snapshots and journal review PDFs.

No fit, source recording, private state, new target or reserved outcome is read.
The public plot coordinates and published predictions are copied unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

import fitz

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'paper/submission_packages_2026-10-16'
OLD = REPO / 'paper/exploratory_evaluation_2026-10-15'
DATA_DIR = REPO / 'results/development'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def check_preservation():
    for name, expected in read(OLD / 'build_receipt.json')['bindings'].items():
        assert sha(REPO / name) == expected, name
    # The administrative preparation record stays local. Public clones still
    # verify the original build receipt and every preserved archive binding.
    preparation = BASE / 'preparation_manifest.json'
    if preparation.is_file():
        record = read(preparation)
        for name, expected in record['input_bindings'].items():
            if name == 'paper/submission_packages_2026-10-16/author_inputs.json':
                # Explicitly recorded author corrections update administration,
                # while all original scientific bindings stay immutable.
                expected = record.get('current_author_input_sha256', expected)
            assert sha(REPO / name) == expected, name
    archive = REPO / 'docs/paper_archive/2026-10-16-before-venue-packages'
    for item in read(archive / 'manifest.json')['files']:
        assert sha(REPO / item['source']) == sha(archive / item['archive']) == item['sha256']


def figures():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    from PIL import Image
    for name in ('arial.ttf', 'arialbd.ttf', 'ariali.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    evidence = read(REPO / 'paper/exploratory_evaluation_2026-10-12/evidence.json')
    panels = [p for p in evidence['pairing_panels'] if p['arm'] == 'validation_unseen']
    anchors = ('joint_spectral', 'eegnet_residual')
    schemes = ('person_same_film_acquisition', 'film_same_person_coarse_valence')
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.25), sharex=True, sharey=True)
    points = []
    colors = ['#2878a5', '#b06b22', '#526359']
    for ai, anchor in enumerate(anchors):
        for si, scheme in enumerate(schemes):
            ax = axes[ai, si]
            title = 'Spectral + context' if ai == 0 else 'EEGNet + context residual'
            relation = 'Other person; same film/acquisition' if si == 0 else 'Other film; same person/coarse valence'
            ax.set_title(title + '\n' + relation, fontsize=8.1, pad=7, linespacing=1.3)
            ax.axvline(0, color='.5', lw=.8, ls='--')
            for rotation in range(3):
                p = next(p for p in panels if (p['anchor'],p['scheme'],p['rotation']) == (anchor,scheme,rotation))
                value = p['mean_full_delta']
                low, high = p['full_panel_crossed_interval']['descriptive_95_percent_interval']
                ax.errorbar(value, 2-rotation, xerr=[[value-low],[high-value]], fmt='o',
                            color=colors[rotation], capsize=2.5, ms=4, lw=1.3)
                points.append({'anchor':anchor,'scheme':scheme,'rotation':rotation,
                               'estimate':value,'interval':[low,high]})
            ax.set_yticks([2,1,0],['R0','R1','R2'],fontsize=8.5)
            ax.tick_params(axis='y',length=0,pad=5)
            ax.tick_params(axis='x',labelsize=8.5,length=3)
            ax.set_xlim(-.27,.375); ax.set_ylim(-.5,2.5)
            ax.set_xticks([-.2,-.1,0,.1,.2,.3]); ax.grid(axis='x',alpha=.15)
            if ai == 1:
                ax.set_xlabel('Swap loss increase (rating-scale MAE)',fontsize=8.2,labelpad=5)
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.875,hspace=.65,wspace=.24)
    fig.text(.5,.08,'Unfamiliar films; mean of 32 repeat losses',ha='center',fontsize=8.1,color='.25')
    fig.text(.5,.035,'Bars: descriptive 95% crossed person/title intervals',ha='center',fontsize=8.1,color='.25')
    shared = BASE / 'shared'; shared.mkdir(exist_ok=True)
    fig.savefig(shared / 'Fig1.pdf')
    fig.savefig(shared / 'Fig1.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    img = Image.open(shared / 'Fig1.tif')
    assert img.size == (3900,2550) and img.mode in ('RGB','RGBA')
    if img.mode == 'RGBA':
        img.convert('RGB').save(shared/'Fig1_rgb.tif',compression='tiff_lzw',dpi=(600,600))
        os.replace(shared/'Fig1_rgb.tif',shared/'Fig1.tif')
    assert (shared/'Fig1.tif').stat().st_size < 10_000_000
    for journal in ('plos_one','mbec'):
        shutil.copy2(shared/'Fig1.tif',BASE/journal/'Fig1.tif')
        shutil.copy2(shared/'Fig1.pdf',BASE/journal/'Fig1.pdf')
    oldpoints = read(REPO/'paper/exploratory_evaluation_2026-10-12/figures/figure_layout_receipt.json')['display_points']
    for p in points:
        q = next(q for q in oldpoints if all(q[k]==p[k] for k in ('anchor','scheme','rotation')))
        assert p['estimate'] == q['mean_full_delta']
        assert p['interval'] == q['descriptive_95_percent_interval']
    nn = read(DATA_DIR/'faced_stronger_baseline_continuation_2026-10-15/comparison.json')
    final = nn['metrics'][-1]
    values = [final['category_held_mae'],final['context_held_mae'],final['ensemble_held_mae']]
    # Native-size vector graphical abstract: one explicitly labeled source cell.
    plt.rcParams.update({'font.size':7.2})
    fig = plt.figure(figsize=(32.93/25.4,37.63/25.4))
    ax = fig.add_axes([.38,.29,.47,.43])
    ax.barh([2,1,0],values,color=['#526359','#2878a5','#b06b22'],height=.5)
    ax.set_yticks([2,1,0],['Category','Context','+ EEG'],fontsize=7.2)
    ax.tick_params(axis='y',length=0,pad=2)
    ax.set_xlim(0,1.8); ax.set_xticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    for y,v in zip([2,1,0],values): ax.text(v+.04,y,f'{v:.2f}',va='center',fontsize=7.2)
    fig.text(.5,.91,'One source cell',ha='center',fontsize=8,weight='bold')
    fig.text(.5,.80,'Held MAE (0-7)',ha='center',fontsize=7.5)
    fig.text(.5,.19,'84 held trials',ha='center',fontsize=7.5)
    fig.text(.5,.09,'Exploratory; lower is better',ha='center',fontsize=6.4)
    fig.savefig(BASE/'mbec/Graphical_Abstract_Muntazar.pdf')
    fig.savefig(BASE/'mbec/Graphical_Abstract_Muntazar.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    caption = ('One reused source-development cell, not independent confirmation: category medians, '
               'matched context and the fixed 360-epoch two-initialization EEGNet/context ensemble '
               'have held mean absolute errors 1.0994, 1.2953 and 1.4075 on 84 trials. '
               'Lower is better. Training and held people/titles are disjoint within this cell. '
               'The graphical abstract summarizes this diagnostic rather than all evaluations. '
               'AI assistance supported plotting-code preparation; values were checked against '
               'the public comparison export.\n')
    write(BASE/'mbec/Graphical_Abstract_Caption.txt',caption)
    result = {'date':'2026-10-16','font_family':'Arial','figure_dimensions_inches':[6.5,4.25],
              'minimum_pairing_text_pt':8.1,'figure_dpi':600,'figure_rgb_no_alpha':True,
              'plotted_points':points,'old_points_identical':True,
              'graphical_abstract_mm':[32.93,37.63],'graphical_abstract_values':values,
              'new_fits':0,'original_or_reserved_data_access':False}
    write(BASE/'figure_export_receipt.json',json.dumps(result,indent=2)+'\n')


def zip_snapshot(path, files, readme):
    bindings = {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(files))}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('README.txt',readme)
        z.writestr('manifest.json',json.dumps({'date':'2026-10-16','files':bindings,
                    'new_fits':0,'reserved_outcomes_included':False},indent=2)+'\n')
        for name in bindings: z.write(REPO/name,name)
    with zipfile.ZipFile(path) as z:
        for name,expected in bindings.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    return {'sha256':sha(path),'bytes':path.stat().st_size,'files':len(bindings)}


def support():
    shared=BASE/'shared';shared.mkdir(exist_ok=True)
    names = [
      'paper/exploratory_evaluation_2026-10-12/evidence.json',
      'paper/exploratory_evaluation_2026-10-12/independent_numeric_audit.json',
      'results/development/faced_inference_quality_2026-10-11/predictions.csv',
      'results/development/faced_category_supported_context_2026-10-12/predictions.csv',
      'results/development/faced_category_supported_context_2026-10-12/comparison.json',
      'results/development/faced_category_supported_context_2026-10-12/source_selection.json',
      'results/development/faced_target_context_pairing_2026-10-12/comparison.json',
      'results/development/faced_target_context_pairing_2026-10-12/target_decomposition.json',
      'results/development/faced_coverage_missing_2026-10-13/cv_predictions.json',
      'results/development/faced_coverage_missing_2026-10-13/comparison.json',
      'results/development/faced_coverage_missing_2026-10-13/independent_audit.json',
      'results/development/faced_stronger_baseline_continuation_2026-10-15/predictions.csv',
      'results/development/faced_stronger_baseline_continuation_2026-10-15/comparison.json',
      'results/development/faced_stronger_baseline_continuation_2026-10-15/histories.json',
      'results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json',
      'paper/exploratory_evaluation_2026-10-15/independent_numeric_review.json']
    datafiles=[REPO/n for n in names]
    data=zip_snapshot(shared/'Supporting_Data.zip',datafiles,
      'Public analysis-level evidence snapshot for the exploratory FACED article.\n'
      'The earlier validation, ten-cell spectral source study and one-cell neural source '
      'diagnostic are different evaluations; do not pool their scores or call them '
      'independent replications. Files retain repository paths for traceability.\n'
      'These are already public development exports, copied byte-for-byte. No raw EEG, '
      'original rating archive, private model state or reserved outcome is included. '
      'Access permissions for original/derived third-party data must be verified before '
      'actual journal submission. This ZIP is not a new dataset or a new fit.\n')
    # Scientific source only; exclude manuscript/administrative builders and code execution.
    codefiles=list((REPO/'gruxnet').glob('*.py'))
    codefiles += [p for p in (REPO/'scripts').glob('*.py')
                  if 'faced' in p.name and 'submission' not in p.name and 'paper' not in p.name]
    codefiles += [p for p in (REPO/'tests').glob('test_faced*.py')]
    codefiles += [REPO/'requirements-publication.txt',REPO/'LICENSE']
    code=zip_snapshot(shared/'Supporting_Code.zip',codefiles,
      'Public FACED analysis/qualification code snapshot.\n'
      'Reported study evidence maps to the data ZIP and repository source paths. '
      'The collection also retains historical diagnostics and inactive guarded '
      'confirmation implementations, which are code rather than completed protected '
      'results. It does not include original recordings/ratings, private checkpoints, '
      'credentials or reserved outcomes, and executes nothing on extraction.\n'
      'Rebuilding raw-data analyses requires independently obtaining FACED through '
      'its original access process, installing documented dependencies and respecting '
      'the source/validation/reserved contracts. Do not launch a protected runner '
      'because this code is present. Full versioned context is in the public repository.\n')
    for journal,aliases in [('plos_one',('S2_Data.zip','S3_Code.zip')),('mbec',('ESM_2.zip','ESM_3.zip'))]:
        for source,target in zip(('Supporting_Data.zip','Supporting_Code.zip'),aliases):
            shutil.copy2(shared/source,BASE/journal/target)
    for item in (data,code): assert item['bytes'] < 10_000_000
    write(BASE/'supporting_snapshot_receipt.json',json.dumps({'data':data,'code':code,
          'all_copied_bytes_verified':True,'original_or_reserved_data_opened':False},indent=2)+'\n')


def letter_tex(text):
    def escape(s):
        for a,b in [('&',r'\&'),('_',r'\_'),('%',r'\%'),('#',r'\#')]: s=s.replace(a,b)
        return s
    return r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=21mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,xurl,hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{2em}\raggedright\pagestyle{empty}
\begin{document}
''' + '\n\n'.join(escape(p).replace('\n',r'\newline ' ) for p in text.strip().split('\n\n')) + '\n'+r'\end{document}'+'\n'


def compile_tex(folder, stem):
    env=dict(os.environ)
    env['TECTONIC_CACHE_DIR']=str(Path(tempfile.gettempdir())/'gru-xnet-tex-cache')
    build=folder/'build';build.mkdir(exist_ok=True)
    subprocess.run([str(REPO/'.publication_deps/tectonic/tectonic.exe'),'--untrusted',
       '--keep-logs','--keep-intermediates','--outdir',str(build),str(folder/(stem+'.tex'))],
       cwd=REPO,env=env,check=True)
    shutil.copy2(build/(stem+'.pdf'),folder/(stem+'.pdf'))


def pdfs(include_administration=False):
    outputs={}
    for journal in ('plos_one','mbec'):
        folder=BASE/journal
        stems = ['manuscript', 'supplementary_methods']
        if include_administration:
            write(folder/'cover_letter.tex',letter_tex((folder/'cover_letter.txt').read_text()))
            stems.append('cover_letter')
        for stem in stems:
            compile_tex(folder,stem)
        final_supp='S1_Appendix.pdf' if journal=='plos_one' else 'ESM_1.pdf'
        shutil.copy2(folder/'supplementary_methods.pdf',folder/final_supp)
        # The actual title page is embedded in each manuscript; this extraction is optional.
        doc=fitz.open(folder/'manuscript.pdf');title=fitz.open();title.insert_pdf(doc,from_page=0,to_page=0)
        title.save(folder/'title_page.pdf');title.close();doc.close()
        outputs[journal]={}
        for stem in stems:
            path=folder/(stem+'.pdf');doc=fitz.open(path)
            outputs[journal][stem]={'pages':len(doc),'bytes':path.stat().st_size,'sha256':sha(path)}
            out=BASE/'tmp/pdfs'/journal/stem;out.mkdir(parents=True,exist_ok=True)
            for i,page in enumerate(doc,1): page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(out/f'page-{i}.png')
            write(out/'extracted.txt','\n'.join(page.get_text() for page in doc))
    write(BASE/'tmp/build_preflight.json',json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-administration', action='store_true',
                        help='Also build local, unpublished cover letters if available.')
    args = parser.parse_args()
    assert not (BASE/'package_receipt.json').exists(),'Preserve sealed package'
    check_preservation()
    figures();support();pdfs(include_administration=args.include_administration)
    check_preservation()


if __name__=='__main__':main()
