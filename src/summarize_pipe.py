import json,glob,re,collections,statistics as st,jiwer,csv
import os;VALID=set(json.load(open('results/valid_ids.json'))) if os.environ.get('VALID') else set(range(1000))
on={}
for f in glob.glob('data/mfa_aligned_v2/*.json'):
    sid=int(re.search(r'(\d+)\.json',f).group(1));a=json.load(open(f))['tiers']['words']['entries']
    on[sid]=a
meta={m['stream_id']:m['keyword_label'] for m in json.load(open('data/streaming_eval/metadata.json'))}
on={s:[e[0] for e in on[s] if e[2].lower()==meta[s]][0] for s in on}
G=collections.defaultdict(list)
for f in glob.glob('results/pipe_*_s*.json'):
    t,p=re.match(r'.*pipe_(.+)_p([0-9.]+)_s\d+\.json',f).groups();G[(t,float(p))]+=[r for r in json.load(open(f)) if r['id'] in VALID]
rows=[]
for (t,p),o in sorted(G.items()):
    D=[r for r in o if r['det'] is not None];V=[r for r in D if r['det']>=on[r['id']]]
    hit=lambda r:r['kw'] in r['hyp'].split()
    rows.append(dict(cond=t,preroll=p,n=len(o),det=len(D)/len(o),early=(len(D)-len(V))/len(o),
      e2e_recall=sum(hit(r) for r in D)/len(o),hit_valid=sum(hit(r) for r in V)/max(1,len(V)),
      wer_kw=jiwer.wer([r['kw'] for r in D],[r['hyp'] or '' for r in D]),
      kws_lat=st.mean(r['det']-on[r['id']] for r in V),asr_wall=st.mean(r['wall'] for r in D),
      e2e_lat=st.mean(r['det']-on[r['id']]+r['wall'] for r in V)))
w=csv.DictWriter(open('results/pipe_summary'+os.environ.get('VALID','')+'.csv','w'),fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(len(rows),'rows')
for r in sorted(rows,key=lambda r:-r['e2e_recall'])[:15]:print(' '.join('%s'%(round(v,3) if isinstance(v,float) else v) for v in r.values()))
