import json,glob,re,collections,csv,numpy as np
meta={m['stream_id']:m['keyword_label'] for m in json.load(open('data/streaming_eval/metadata.json'))};on={}
for f in glob.glob('data/mfa_aligned_v2/*.json'):
    s=int(re.search(r'(\d+)\.json',f).group(1));on[s]=[e[0] for e in json.load(open(f))['tiers']['words']['entries'] if e[2].lower()==meta[s]][0]
V=set(json.load(open('results/valid_ids.json')));G=collections.defaultdict(list)
for f in glob.glob('results/pipes_*_s*.json'):
    t,p=re.match(r'.*pipes_(.+)_p([0-9.]+)_s\d+\.json',f).groups();G[(t,float(p))]+=json.load(open(f))
rows=[]
for sub,ids in (('valid',V),('all',set(range(1000)))):
    for (t,p),o in sorted(G.items()):
        o=[r for r in o if r['id'] in ids];F=[r for r in o if r['found_t'] is not None]
        L=np.array([r['found_t']-on[r['id']] for r in F if r['found_t']>=on[r['id']]]) if F else np.array([np.nan])
        rows.append(dict(subset=sub,cond=t,preroll=p,n=len(o),recall=len(F)/len(o),lat_mean=np.nanmean(L),lat_p50=np.nanmedian(L),lat_p95=np.nanpercentile(L,95)))
w=csv.DictWriter(open('results/pipes_summary.csv','w'),fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for r in sorted([r for r in rows if r['subset']=='valid'],key=lambda r:-r['recall'])[:12]:print(' '.join(str(round(v,3)) if isinstance(v,float) else str(v) for v in r.values()))
