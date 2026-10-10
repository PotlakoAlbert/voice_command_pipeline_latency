import json,sys,time,torch,torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH as B
lat,P,K,N,tag=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
torch.set_num_threads(4)
fe=B.get_streaming_feature_extractor();dec=B.get_decoder();tp=B.get_token_processor();dev='cuda';dec.model.to(dev)
H=B.hop_length;seg=B.segment_length*H;ctx=B.right_context_length*H;SR=16000
L=json.load(open('results/'+lat))
L=L if isinstance(L,list) else L.get('results',L)
det={e['stream_id']:e.get('detection_time') for e in L}
meta=json.load(open('data/streaming_eval/metadata.json'))[K::N]
out=[]
for m in meta:
    sid=m['stream_id'];d=det.get(sid);r=dict(id=sid,kw=m['keyword_label'],det=d,hyp=None,wall=None)
    if d is not None:
        w,sr=torchaudio.load('data/streaming_eval/audio/'+m['filename']);w=w[0]
        a=max(0,int((d-P)*SR));b=min(len(w),int((d+1.0)*SR));w=w[a:b]
        state=hyp=None;cache=torch.zeros(ctx);t0=time.time()
        with torch.no_grad():
            for s in range(0,len(w),seg):
                c=w[s:s+seg]
                if len(c)<seg:c=torch.nn.functional.pad(c,(0,seg-len(c)))
                x=torch.cat((cache,c));cache=c[-ctx:]
                f,l=fe(x);hy,state=dec.infer(f.to(dev),l.to(dev),10,state=state,hypothesis=hyp);hyp=hy
        torch.cuda.synchronize();r['wall']=time.time()-t0;r['hyp']=tp(hyp[0][0],lstrip=False).lower().strip()
    out.append(r)
json.dump(out,open('results/pipe_%s_p%s_s%d.json'%(tag,P,K),'w'))
print('done',K,len(out))
