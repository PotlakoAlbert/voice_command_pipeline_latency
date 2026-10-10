import json,sys,time,torch,torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH as B
lat,P,K,N,tag=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
fe=B.get_streaming_feature_extractor();dec=B.get_decoder();tp=B.get_token_processor();dev='cuda';dec.model.to(dev)
H=B.hop_length;seg=B.segment_length*H;ctx=B.right_context_length*H;SR=16000
L=json.load(open('results/'+lat));L=L if isinstance(L,list) else L.get('results',L)
det={e['stream_id']:e.get('detection_time') for e in L}
meta=json.load(open('data/streaming_eval/metadata.json'))[K::N];out=[]
for m in meta:
    sid=m['stream_id'];d=det.get(sid);r=dict(id=sid,kw=m['keyword_label'],det=d,found_t=None,hyp=None)
    if d is not None:
        w=torchaudio.load('data/streaming_eval/audio/'+m['filename'])[0][0]
        a=max(0,int((d-P)*SR));b=min(len(w),int((d+1.0)*SR))
        state=hyp=None;cache=torch.zeros(ctx);clock=d;txt=''
        with torch.no_grad():
            for s in range(a,b,seg):
                c=w[s:min(s+seg,b)];te=min(s+seg,b)/SR
                if len(c)<seg:c=torch.nn.functional.pad(c,(0,seg-len(c)))
                start=max(clock,te if te>d else d);torch.cuda.synchronize();t0=time.perf_counter()
                x=torch.cat((cache,c));cache=c[-ctx:];f,l=fe(x)
                hy,state=dec.infer(f.to(dev),l.to(dev),10,state=state,hypothesis=hyp);hyp=hy
                torch.cuda.synchronize();clock=start+(time.perf_counter()-t0)
                try:txt=tp(hyp[0][0],lstrip=False).lower().strip()
                except Exception:txt=''
                if r['found_t'] is None and m['keyword_label'] in txt.split():r['found_t']=clock
        r['hyp']=txt
    out.append(r)
json.dump(out,open('results/pipes_%s_p%s_s%d.json'%(tag,P,K),'w'));print('done',K,len(out))
