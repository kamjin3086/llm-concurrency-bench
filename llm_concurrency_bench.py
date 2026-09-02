#!/usr/bin/env python3
"""Zero-dependency concurrency benchmark for OpenAI-compatible llama-server endpoints."""
from __future__ import annotations
import argparse, csv, json, math, statistics, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class Result:
    ok: bool; worker: int; started: float; first: float|None; finished: float
    prompt_tokens: int; completion_tokens: int; finish_reason: str|None; error: str|None=None
    @property
    def ttft(self): return None if self.first is None else self.first-self.started
    @property
    def decode_s(self): return None if self.first is None else max(1e-9,self.finished-self.first)
    @property
    def stream_tps(self): return None if not self.completion_tokens or self.decode_s is None else self.completion_tokens/self.decode_s

def pct(v,p):
    if not v: return float('nan')
    v=sorted(v)
    if len(v)==1:return v[0]
    x=(len(v)-1)*p; lo=math.floor(x); hi=math.ceil(x)
    return v[lo] if lo==hi else v[lo]*(hi-x)+v[hi]*(x-lo)

def get_json(url, timeout=10):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept':'application/json'}),timeout=timeout) as r:
        return json.loads(r.read().decode())

def model_id(base, configured=None):
    if configured:return configured
    urls=[base.rstrip('/')+'/models'] if base.rstrip('/').endswith('/v1') else [base.rstrip('/')+'/v1/models',base.rstrip('/')+'/models']
    err=None
    for u in urls:
        try:
            d=get_json(u); items=d.get('data',[])
            if items:return items[0]['id']
        except Exception as e: err=e
    raise RuntimeError(f'cannot discover model id; set profile.model. last error={err}')

def endpoint(base):
    b=base.rstrip('/'); return b+'/chat/completions' if b.endswith('/v1') else b+'/v1/chat/completions'

def has_token(chunk):
    for c in chunk.get('choices') or []:
        d=c.get('delta') or {}
        if any(isinstance(d.get(k),str) and d.get(k) for k in ('content','reasoning_content','reasoning')): return True
        if d.get('tool_calls'): return True
    return False

def request_one(ep, model, profile, cfg, worker, gate):
    timeout=float(profile.get('timeout',cfg.get('timeout',900)))
    max_tokens=int(profile.get('max_tokens',cfg.get('max_tokens',512)))
    temp=float(profile.get('temperature',cfg.get('temperature',0.0)))
    seed=int(profile.get('seed',cfg.get('seed',42)))
    system=profile.get('system_prompt',cfg.get('system_prompt','You are a senior software engineer working as one worker in a parallel coding-agent team.'))
    prompt=profile.get('prompt',cfg.get('prompt','Design and explain a production-quality implementation for a moderately complex software component. Include architecture, data structures, concurrency/error handling, edge cases, testing strategy, and representative code. Continue in technical detail until the token budget is exhausted.'))
    if profile.get('unique_prompts',cfg.get('unique_prompts',True)):
        prompt += f'\n\nYou are worker #{worker}. Use scenario variant #{worker} with a distinct module name and independent implementation details.'
    body={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}], 'stream':True,'stream_options':{'include_usage':True},'max_tokens':max_tokens,'temperature':temp,'seed':seed+worker}
    extra={}; extra.update(cfg.get('extra_body',{})); extra.update(profile.get('extra_body',{})); body.update(extra)
    req=urllib.request.Request(ep,data=json.dumps(body,ensure_ascii=False).encode(),method='POST',headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    try: gate.wait(timeout=30)
    except threading.BrokenBarrierError: pass
    started=time.perf_counter(); first=None; pt=ct=0; reason=None
    try:
        with urllib.request.urlopen(req,timeout=timeout) as resp:
            for raw in resp:
                line=raw.decode('utf-8','replace').strip()
                if not line.startswith('data:'): continue
                s=line[5:].strip()
                if s=='[DONE]': break
                try: chunk=json.loads(s)
                except json.JSONDecodeError: continue
                if first is None and has_token(chunk): first=time.perf_counter()
                for c in chunk.get('choices') or []:
                    if c.get('finish_reason') is not None: reason=c.get('finish_reason')
                u=chunk.get('usage')
                if u:
                    pt=int(u.get('prompt_tokens') or pt or 0); ct=int(u.get('completion_tokens') or ct or 0)
        return Result(True,worker,started,first,time.perf_counter(),pt,ct,reason)
    except Exception as e:
        return Result(False,worker,started,None,time.perf_counter(),0,0,None,repr(e))


def preload_model(profile, cfg, model, ep):
    """Trigger llama-swap loading and wait for one REAL successful generation.
    This entire phase is unmeasured and never enters benchmark results.
    """
    if not profile.get('preload', cfg.get('preload', True)):
        return

    timeout = float(profile.get('preload_timeout', cfg.get('preload_timeout', 1800)))
    retry = float(profile.get('preload_retry_interval', cfg.get('preload_retry_interval', 3)))
    deadline = time.monotonic() + timeout
    attempt = 0

    temp_profile = dict(profile)
    temp_profile['max_tokens'] = int(profile.get('preload_max_tokens', cfg.get('preload_max_tokens', 8)))
    temp_profile['unique_prompts'] = False
    temp_profile['system_prompt'] = 'You are a concise assistant.'
    temp_profile['prompt'] = profile.get('preload_prompt', cfg.get('preload_prompt', 'Reply with exactly: READY'))
    temp_profile['temperature'] = 0.0
    temp_profile['timeout'] = timeout

    print('  preload (unmeasured): waiting for llama-swap/model to return a real completion ...', flush=True)

    while True:
        attempt += 1
        r = request_one(ep, model, temp_profile, cfg, 1, threading.Barrier(1))
        if r.ok and r.first is not None:
            print(f'    READY after {r.finished-r.started:.2f}s (TTFT={r.ttft:.2f}s). Startup time discarded.')
            settle = float(profile.get('settle_seconds', cfg.get('settle_seconds', 1.0)))
            if settle > 0:
                time.sleep(settle)
            return

        if time.monotonic() >= deadline:
            raise RuntimeError(f'model did not become ready within {timeout:.0f}s; last error={r.error}')

        print(f'    preload attempt {attempt} not ready: {r.error or "no generated token"}; retry in {retry:.1f}s', flush=True)
        time.sleep(retry)

def run_batch(profile,cfg,c,model,ep):
    gate=threading.Barrier(c); rs=[]
    with ThreadPoolExecutor(max_workers=c) as ex:
        fs=[ex.submit(request_one,ep,model,profile,cfg,i+1,gate) for i in range(c)]
        for f in as_completed(fs): rs.append(f.result())
    ok=[r for r in rs if r.ok]
    if not ok: raise RuntimeError('; '.join(r.error or '?' for r in rs))
    start=min(r.started for r in ok); end=max(r.finished for r in ok); wall=max(1e-9,end-start)
    firsts=[r.first for r in ok if r.first is not None]; total=sum(r.completion_tokens for r in ok); prompts=sum(r.prompt_tokens for r in ok)
    tt=[r.ttft for r in ok if r.ttft is not None]; st=[r.stream_tps for r in ok if r.stream_tps is not None]
    decode_window=max(1e-9,end-min(firsts)) if firsts else float('nan')
    return {'profile':profile['name'],'concurrency':c,'ok_requests':len(ok),'failed_requests':len(rs)-len(ok),'prompt_tokens':prompts,'completion_tokens':total,'wall_s':wall,'aggregate_e2e_tps':total/wall if total else 0.0,'aggregate_decode_tps':total/decode_window if firsts and total else 0.0,'avg_stream_decode_tps':statistics.mean(st) if st else float('nan'),'ttft_avg_s':statistics.mean(tt) if tt else float('nan'),'ttft_p50_s':pct(tt,.5) if tt else float('nan'),'ttft_max_s':max(tt) if tt else float('nan'),'finish_reasons':','.join(str(r.finish_reason) for r in ok),'errors':' | '.join(r.error or '' for r in rs if not r.ok)}

def f(v):
    return 'n/a' if isinstance(v,float) and math.isnan(v) else (f'{v:.2f}' if isinstance(v,float) else str(v))

def print_rows(rows):
    h=['profile','c','tok','wall','agg e2e','agg decode','avg stream','TTFT p50','TTFT max','fail']
    t=[[r['profile'],str(r['concurrency']),str(r['completion_tokens']),f(r['wall_s']),f(r['aggregate_e2e_tps']),f(r['aggregate_decode_tps']),f(r['avg_stream_decode_tps']),f(r['ttft_p50_s']),f(r['ttft_max_s']),str(r['failed_requests'])] for r in rows]
    w=[len(x) for x in h]
    for row in t:
        for i,x in enumerate(row):w[i]=max(w[i],len(x))
    line=lambda row:'  '.join(x.ljust(w[i]) for i,x in enumerate(row))
    print('\n'+line(h));print(line(['-'*x for x in w]));[print(line(r)) for r in t];print()

def main():
    a=argparse.ArgumentParser();a.add_argument('config');a.add_argument('--profile');a.add_argument('--out',default='bench-results.csv');args=a.parse_args()
    cfg=json.loads(Path(args.config).read_text()); profiles=cfg.get('profiles') or []; allrows=[]
    for p in profiles:
        if args.profile and args.profile.lower() not in p['name'].lower():continue
        base=p['base_url']; ep=endpoint(base)
        try:m=model_id(base,p.get('model'))
        except Exception as e: print(f"[{p['name']}] {e}",file=sys.stderr);continue
        raw_cs=p.get('concurrencies',cfg.get('concurrencies')); cs=raw_cs or [1,2,3,4]; mc=p.get('max_concurrency'); cs=[int(x) for x in cs if mc is None or int(x)<=int(mc)]
        warm=int(p.get('warmup',cfg.get('warmup',1))); reps=int(p.get('repetitions',cfg.get('repetitions',2)))
        print(f"\n=== {p['name']} ===\nendpoint: {ep}\nmodel: {m}\nconcurrency: {cs}")
        try:
            preload_model(p,cfg,m,ep)
        except Exception as e:
            print(f'  PRELOAD FAILED: {e}', file=sys.stderr)
            print('  Skipping this profile.', file=sys.stderr)
            continue
        for i in range(warm):
            print(f'  warmup {i+1}/{warm} ...',flush=True)
            try:run_batch(p,cfg,1,m,ep)
            except Exception as e:print(f'  warmup failed: {e}',file=sys.stderr)
        rows=[]
        for c in cs:
            for rep in range(1,reps+1):
                print(f'  c={c}, run={rep}/{reps} ...',flush=True)
                try:
                    r=run_batch(p,cfg,c,m,ep); r['run']=rep; rows.append(r); allrows.append(r)
                    print(f"    agg_decode={r['aggregate_decode_tps']:.2f} tok/s | agg_e2e={r['aggregate_e2e_tps']:.2f} | avg_stream={r['avg_stream_decode_tps']:.2f} | TTFT_p50={r['ttft_p50_s']:.2f}s")
                except Exception as e:print(f'    FAILED: {e}',file=sys.stderr)
        print_rows(rows)
        print(f"  profile complete: {p['name']} — only now moving to the next model")
    if allrows:
        with open(args.out,'w',newline='',encoding='utf-8') as fp:
            wr=csv.DictWriter(fp,fieldnames=list(allrows[0].keys()));wr.writeheader();wr.writerows(allrows)
        print(f'CSV written to: {args.out}')
        groups={}
        for r in allrows:groups.setdefault((r['profile'],r['concurrency']),[]).append(r)
        summary=[]
        for (name,c),rs in groups.items():
            summary.append({'profile':name,'concurrency':c,'completion_tokens':round(statistics.mean(r['completion_tokens'] for r in rs)),'wall_s':statistics.mean(r['wall_s'] for r in rs),'aggregate_e2e_tps':statistics.mean(r['aggregate_e2e_tps'] for r in rs),'aggregate_decode_tps':statistics.mean(r['aggregate_decode_tps'] for r in rs),'avg_stream_decode_tps':statistics.mean(r['avg_stream_decode_tps'] for r in rs),'ttft_p50_s':statistics.mean(r['ttft_p50_s'] for r in rs),'ttft_max_s':statistics.mean(r['ttft_max_s'] for r in rs),'failed_requests':sum(r['failed_requests'] for r in rs)})
        print('\n=== Mean summary ===');print_rows(summary)
    return 0
if __name__=='__main__':raise SystemExit(main())
