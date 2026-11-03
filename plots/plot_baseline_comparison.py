#!/usr/bin/env python3
"""Compare fixed-only, naive reclamation, and idealized recovery Frontier runs.

Run after simulations finish. Reads each run's native summary and request events;
plots measured-arrival latency P95 by 20-second arrival windows and overall metrics.
Usage:
 python plot_baseline_comparison.py --fixed outputs/malleserve/runs/baseline_fixed4_r10 \
   --naive outputs/malleserve/runs/baseline_naive_r10 \
   --ideal outputs/malleserve/runs/baseline_ideal_r10 \
   --output outputs/malleserve/figures/baseline_comparison
"""
import argparse, csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'Fixed 4 GPUs':'#607086','Naive':'#223C61','Ideal recovery':'#19A398'}

def q95(v):
    return float(np.percentile(v,95)) if v else np.nan

def load(path):
    summary=json.loads((path/'malleserve_summary.json').read_text())
    arrivals={}; completions={}
    with (path/'metrics_ground_truth.jsonl').open() as f:
        for line in f:
            if not line.strip(): continue
            e=json.loads(line); rid=str(e.get('request_id'))
            if e.get('event_type')=='request_arrival':
                if rid in arrivals: raise ValueError(f'Duplicate arrival: {rid}')
                arrivals[rid]=float(e['arrived_at'])
            elif e.get('event_type')=='request_completion':
                if rid in completions: raise ValueError(f'Duplicate completion: {rid}')
                completions[rid]=float(e['completed_at'])
    lifecycle_path=path/'transient_lifecycle.json'
    life=json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else {}
    return summary,arrivals,completions,life

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    print('Saved',path.with_suffix('.png'),path.with_suffix('.pdf'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixed','naive','ideal'): p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--window-seconds',type=float,default=20)
    a=p.parse_args()
    paths={'Fixed 4 GPUs':a.fixed,'Naive':a.naive,'Ideal recovery':a.ideal}
    runs={name:load(path) for name,path in paths.items()}
    windows=[]
    for name,(s,arr,done,life) in runs.items():
        start,end=map(float,s['measurement_window_s'])
        windows.append((start,end))
        if s['measured_no_completion_event']:
            print('WARNING: incomplete measured requests:',name,s['measured_no_completion_event'])
    if len(set(windows))!=1: raise ValueError(f'Different measurement windows: {windows}')
    ref_arrivals=list(runs['Fixed 4 GPUs'][1].values())
    for name,(_,arr,_,_) in runs.items():
        if len(arr)!=len(ref_arrivals) or not np.allclose(sorted(arr.values()),sorted(ref_arrivals),atol=1e-8):
            raise ValueError(f'Arrival schedules do not match: {name}')
    start,end=windows[0]
    edges=np.arange(start,end+a.window_seconds,a.window_seconds)
    if edges[-1]<end-1e-7: edges=np.append(edges,end)
    edges[-1]=end
    mids=(edges[:-1]+edges[1:])/2
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.6,4.3));fig.subplots_adjust(left=.10,right=.97,top=.84,bottom=.16)
    rows=[]
    for name,(s,arr,done,life) in runs.items():
        vals=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            ids=[rid for rid,t in arr.items() if lo<=t<hi]
            latencies=[done[rid]-arr[rid] for rid in ids if rid in done]
            if any(t<0 for t in latencies): raise ValueError('Negative latency')
            vals.append(q95(latencies))
        ax.plot(mids,vals,'o-',lw=2,ms=4.5,label=name,color=COLORS[name])
        rows.append({'policy':name,'p95_s':s.get('latency_p95_s_for_completed_measured'),
                     'slo_attainment':s.get('measured_slo_attainment'),
                     'goodput_rps':s.get('measured_slo_goodput_rps_allowing_drain'),
                     'completed':s.get('measured_completed_eventually'),
                     'measured_arrivals':s.get('measured_arrivals'),
                     'retries':life.get('retries',0),
                     'lost_processed_tokens':life.get('lost_processed_tokens',0),
                     'oracle_restored_tokens':life.get('oracle_restored_tokens',0)})
    ax.set_title('Fixed vs naive vs ideal recovery | request latency',fontsize=11.5,fontweight='bold',color='#18253A',pad=12)
    ax.set_xlabel('Request arrival time (s)');ax.set_ylabel('P95 latency within arrival window (s)')
    ax.grid(axis='y',color='#E8EDF3');ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.09),fontsize=9)
    save(fig,a.output.with_name(a.output.name+'_p95_timeline'))
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6));fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.22,wspace=.28)
    names=list(paths); x=np.arange(3)
    for ax,key,label,unit in [(axes[0],'p95_s','Measured P95 latency','s'),(axes[1],'slo_attainment','SLO attainment','%')]:
        vals=[float(r[key])*(100 if unit=='%' else 1) if r[key] is not None else np.nan for r in rows]
        bars=ax.bar(x,vals,color=[COLORS[n] for n in names],width=.58)
        ax.set_xticks(x,['Fixed-only','Naive','Ideal recovery'])
        ax.set_title(label,fontweight='semibold',loc='left',fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0,max(v for v in vals if np.isfinite(v))*1.22 if any(np.isfinite(vals)) else 1)
        ax.grid(axis='y',color='#E8EDF3');ax.set_axisbelow(True)
        for b,v in zip(bars,vals):
            if np.isfinite(v):ax.annotate(f'{v:.2f}' if unit=='s' else f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    save(fig,a.output.with_name(a.output.name+'_summary'))
    out=a.output.with_name(a.output.name+'_summary.csv');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved',out)

if __name__=='__main__': main()
