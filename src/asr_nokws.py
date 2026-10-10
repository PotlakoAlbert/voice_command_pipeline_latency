import json,sys,torch,torchaudio,jiwer
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH as B
K,N=int(sys.argv[1]),int(sys.argv[2])
torch.set_num_threads(4)
fe=B.get_streaming_feature_extractor();dec=B.get_decoder();tp=B.get_token_processor()
H=B.hop_length;seg=B.segment_length*H;ctx=B.right_context_length*H
meta=json.load(open('data/streaming_eval/metadata.json'))[K::N]
out=[];R=[];Y=[]
for m in meta:
    w,sr=torchaudio.load('data/streaming_eval/audio/'+m['filename']);assert sr==16000;w=w[0]
    a=json.load(open('data/mfa_aligned_v2/stream_%05d.json'%m['stream_id']))
    ref=' '.join(e[2] for e in a['tiers']['words']['entries']).lower()
    state=hyp=None;cache=torch.zeros(ctx)
    with torch.no_grad():
        for s in range(0,len(w),seg):
            c=w[s:s+seg]
            if len(c)<seg:c=torch.nn.functional.pad(c,(0,seg-len(c)))
            x=torch.cat((cache,c));cache=c[-ctx:]
            f,l=fe(x)
            hy,state=dec.infer(f,l,10,state=state,hypothesis=hyp);hyp=hy
    h=tp(hyp[0][0],lstrip=False).lower().strip()
    out.append(dict(id=m['stream_id'],kw=m['keyword_label'],ref=ref,hyp=h));R.append(ref);Y.append(h)
    if len(out)%20==0:print(len(out),ref,'|',h,flush=True)
print('shard',K,'WER',jiwer.wer(R,Y))
json.dump(out,open('results/asr_nokws_s%d.json'%K,'w'))
