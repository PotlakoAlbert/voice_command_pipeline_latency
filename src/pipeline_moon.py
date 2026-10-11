import json,sys,time,math,torch,torchaudio
from transformers import AutoProcessor, MoonshineForConditionalGeneration as M
lat,P,K,N,tag=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
dev='cuda';pr=AutoProcessor.from_pretrained('UsefulSensors/moonshine-tiny');md=M.from_pretrained('UsefulSensors/moonshine-tiny').to(dev).eval();SR=16000
L=json.load(open('results/'+lat));L=L if isinstance(L,list) else L.get('results',L)
det={e['stream_id']:e.get('detection_time') for e in L}
meta=json.load(open('data/streaming_eval/metadata.json'))[K::N];out=[]
def tr(x):
    i=pr(x.numpy(),sampling_rate=SR,return_tensors='pt').to(dev)
    with torch.no_grad():g=md.generate(**i,max_new_tokens=max(4,math.ceil(len(x)/SR*6.5)))
    return pr.batch_decode(g,skip_special_tokens=True)[0].lower().replace(',','').replace('.','').strip()
for m in meta:
    sid=m['stream_id'];d=det.get(sid);r=dict(id=sid,kw=m['keyword_label'],det=d,found_t=None,hyp=None)
    if d is not None:
        w=torchaudio.load('data/streaming_eval/audio/'+m['filename'])[0][0];a=max(0,int((d-P)*SR));txt=''
        for dl in (0.25,0.5,0.75,1.0):
            b=min(len(w),int((d+dl)*SR));torch.cuda.synchronize();t0=time.perf_counter()
            txt=tr(w[a:b]);torch.cuda.synchronize();c=time.perf_counter()-t0
            if m['keyword_label'] in txt.split():r['found_t']=d+dl+c;break
        r['hyp']=txt
    out.append(r)
json.dump(out,open('results/pipem_%s_p%s_s%d.json'%(tag,P,K),'w'));print('done',K,len(out))
