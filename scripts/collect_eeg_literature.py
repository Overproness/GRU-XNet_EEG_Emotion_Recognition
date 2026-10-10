"""Read-only paper metadata watch; scientific novelty assessment stays manual."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_QUERIES = [
    '(all:EEG OR all:electroencephalography) AND all:emotion',
    'all:EEG AND (all:stimulus OR all:confound OR all:generalization)',
]
CROSSREF_QUERIES = ['EEG emotion recognition generalization', 'EEG emotion stimulus confounding']
KNOWN_IDS = ['2510.22197', '2607.04139', '2610.03618', '2412.07236']
NS = {'a':'http://www.w3.org/2005/Atom'}


def fetch(url, limit=2000000):
    request = urllib.request.Request(url, headers={'User-Agent':'GRU-XNet-public-literature-watch/1.0', 'Accept':'application/json, application/atom+xml, application/xml'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(limit+1)
            if len(data)>limit:raise ValueError('Public metadata exceeds budget')
            return data, {'url':url,'http_status':response.status,'bytes':len(data),'body_sha256':hashlib.sha256(data).hexdigest()}
    except urllib.error.HTTPError as error:
        return None, {'url':url,'http_status':error.code,'failure':'HTTPError'}
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None, {'url':url,'http_status':None,'failure':'network_timeout_or_size'}


def parse_arxiv(data):
    root=ET.fromstring(data);records=[]
    if root.tag!='{http://www.w3.org/2005/Atom}feed':raise ValueError('Not an Atom feed')
    for entry in root.findall('a:entry',NS):
        url=entry.findtext('a:id',default='',namespaces=NS)
        if not re.match(r'https?://arxiv\.org/abs/\d{4}\.\d{4,5}(v\d+)?$',url):
            raise ValueError('Invalid paper entry (including API error feeds)')
        records.append({'provider':'arxiv','id':url.rsplit('/',1)[-1], 'url':url.replace('http:','https:',1),
                        'title':' '.join(entry.findtext('a:title',default='',namespaces=NS).split()),
                        'published':entry.findtext('a:published',namespaces=NS),'updated':entry.findtext('a:updated',namespaces=NS),
                        'authors':[a.findtext('a:name',namespaces=NS) for a in entry.findall('a:author',NS)],
                        'review_status':'candidate; no acceptance or methods assessment inferred'})
    return records


def parse_crossref(data):
    root=json.loads(data)
    if root.get('status')!='ok' or not isinstance(root.get('message',{}).get('items'),list):raise ValueError('Invalid Crossref result')
    records=[]
    for item in root['message']['items']:
        doi=item.get('DOI')
        if not doi:continue
        records.append({'provider':'crossref','id':doi.lower(),'url':'https://doi.org/'+doi,
                        'title':' '.join(item.get('title',[])), 'published_online':item.get('published-online',{}).get('date-parts'),
                        'published_print':item.get('published-print',{}).get('date-parts'),'issued':item.get('issued',{}).get('date-parts'),
                        'indexed':item.get('indexed',{}).get('date-time'),
                        'authors':[(' '.join((a.get('given',''),a.get('family','')))).strip() for a in item.get('author',[])],
                        'review_status':'search candidate; inspect primary publication and methods before relying on it'})
    return records


def collect(out, days=90):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cutoff=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=days)).date().isoformat()
    records=[];logs=[]
    endpoints=[]
    for query in ARXIV_QUERIES:
        endpoints.append(('arxiv','https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':query,'start':0,'max_results':30,'sortBy':'lastUpdatedDate','sortOrder':'descending'})))
    endpoints.append(('arxiv','https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'id_list':','.join(KNOWN_IDS)})))
    for query in CROSSREF_QUERIES:
        endpoints.append(('crossref','https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':query,'filter':'from-index-date:'+cutoff,'rows':30,'sort':'indexed','order':'desc','select':'DOI,title,author,published-online,published-print,issued,indexed'})))
    for index,(provider,url) in enumerate(endpoints):
        if index:time.sleep(3.1)
        data,log=fetch(url)
        log['provider']=provider
        if data is not None:
            try:
                parsed=parse_arxiv(data) if provider=='arxiv' else parse_crossref(data)
                records.extend(parsed);log.update({'semantic_success':True,'candidate_count':len(parsed)})
            except (ValueError,ET.ParseError,KeyError,TypeError):
                log.update({'semantic_success':False,'failure':'invalid_metadata_response'})
        else:log['semantic_success']=False
        logs.append(log)
    unique={(r['provider'],r['id']):r for r in records}
    result={'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'timezone':'Asia/Karachi',
            'scope':'Public bibliographic candidates and watched arXiv revisions; no full text, abstracts, data, outcomes or automatic novelty claims.',
            'lookback_days_crossref_indexing':days,'arxiv_search_coverage':'up to 30 most recently updated results per fixed query, not an exhaustive date-bounded search',
            'known_arxiv_ids':KNOWN_IDS, 'retrievals':logs,'provider_successes':{p:sum(r['semantic_success'] for r in logs if r['provider']==p) for p in ('arxiv','crossref')},
            'partial_or_failed':any(not r['semantic_success'] for r in logs),'candidates':sorted(unique.values(),key=lambda r:(r['provider'],r['id'])),'scientific_review_completed':False}
    (out/'candidates.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    lines=['# EEG paper metadata watch','', 'This is a candidate list. Primary-source methods and novelty still require review.', '',
           f"Checked: {result['checked_at_utc']}. Partial/failed retrievals: {result['partial_or_failed']}.", '',
           'Crossref indexing dates are discovery dates; they do not establish when research was published. arXiv publication and revision dates remain separate.', '']
    for record in result['candidates']:
        title=record['title'].replace('[','(').replace(']',')').replace('\n',' ')
        lines.append(f"- [{title}]({record['url']}) — {record['provider']}; methods review pending.")
    (out/'REVIEW_QUEUE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'candidates':len(unique),'provider_successes':result['provider_successes'],'partial_or_failed':result['partial_or_failed']}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);parser.add_argument('--days',type=int,default=90)
    args=parser.parse_args();result=collect(args.out,args.days)
    if not any(result['provider_successes'].values()):raise SystemExit(1)
