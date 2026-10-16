"""Render manuscript tables from saved results without performing inference or training."""
from pathlib import Path
import csv, json, hashlib, html, math, statistics

ROOT=Path(__file__).parent
SUITE=ROOT.parent
VDA=SUITE.parent/'StabilizedAttentionVDA'

def build_results():
    files={
        'parent_test':SUITE/'LocalRuntime/continuation02/final_test.json',
        'vda_test':VDA/'LocalRuntime/continuation01/final_test.json',
        'primary':VDA/'Analysis/run01/primary_summary.json',
        'primary_tests':VDA/'Analysis/run01/primary_hypothesis_tests.csv',
        'native':VDA/'Analysis/run01/fresh_native_results.csv',
        'exploratory':VDA/'Analysis/run01/all_intervention_effects.csv',
        'earlier':SUITE/'Analysis/run01/summary.json',
        'verified':VDA/'Analysis/run01/completion_verified.json',
    }
    data={k:(list(csv.DictReader(p.open(encoding='utf-8'))) if p.suffix=='.csv' else json.loads(p.read_text(encoding='utf-8'))) for k,p in files.items()}
    assert data['verified']['status']=='complete_and_verified'
    def table(headers,rows):
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def pct(x):return f'{100*float(x):.2f}%'
    names={t['id']:t['title'] for t in json.loads((ROOT/'task-atlas/tasks.json').read_text(encoding='utf-8'))}
    parent=data['parent_test']['summary']['tasks']
    cells=data['vda_test']['cells']
    keys=list(parent)+['vda4','vda9','vda16']
    rows=[]
    for k in keys:
        vals=[c['ba'] for c in cells if c['task']==k and c.get('ba') is not None and math.isfinite(c['ba'])]
        rows.append([names.get(k,k.upper()),pct(parent[k]['balanced_accuracy']) if k in parent else 'Not trained',pct(statistics.mean(vals))])
    out={'lineage-results':table(['Task','13-task parent: BA','16-task continuation: BA'],rows)}
    effects={(x['arm'],x['event']):x for x in data['primary']['effects']}
    rows=[]
    for event,label in [('cued','Bottom-right change: hit rate'),('uncued','Top-left change: hit rate')]:
        a=effects['sham',event];b=effects['cued_-4_change',event]
        rows.append([label,pct(a['hit_rate']),pct(b['hit_rate'])])
    rows.append(['No change: false-alarm rate',pct(effects['sham','uncued']['false_alarm_rate']),pct(effects['cued_-4_change','uncued']['false_alarm_rate'])])
    out['primary-rates']=table(['Response measure','Sham','Bottom-right inhibition −4'],rows)
    armnames={'cued_-4_change':'Inhibit cued −4','uncued_+4_change':'Stimulate uncued +4','cued_+4_change':'Stimulate cued +4','uncued_-4_change':'Inhibit uncued −4'}
    out['primary-tests']=table(['Manipulation','Event site / hypothesis','Hit difference (points)','Holm-adjusted p'],[[armnames[x['arm']],x['event']+' / '+x['hypothesis'],f"{100*float(x['delta']):+.2f}",f"{float(x['p_holm']):.3g}"] for x in data['primary_tests']])
    out['native-vda-results']=table(['Task','Maximum rotation','Trials','Balanced accuracy','AUC','Hits','Catch false alarms'],[['VDA'+x['size'],x['theta']+'°',x['n'],pct(x['ba']),f"{float(x['auc']):.3f}",pct(x['hit_rate']),pct(x['false_alarm_rate'])] for x in data['native']])
    selected=[x for x in data['exploratory'] if float(x['validity'])==1 and float(x['angle'])==18 and x['arm']=='cued_-4_change']
    assert len(selected)==6
    out['exploratory-sizes']=table(['Task','Event site','Hit difference (points)','BA difference (points)','BA 95% interval (points)'],[['VDA'+x['size'],x['event'],f"{100*float(x['hit_delta']):+.2f}",f"{100*float(x['ba_delta']):+.2f}",f"{100*float(x['low']):+.2f} to {100*float(x['high']):+.2f}"] for x in selected])
    out['earlier-benefits']=table(['Original-suite task','Intervention','BA difference (points)','95% interval (points)','Holm p'],[[names[x['task']],x['arm'].replace('_',' '),f"{x['delta_pp']:+.2f}",f"{x['low_pp']:+.2f} to {x['high_pp']:+.2f}",f"{x['p_holm']:.3g}"] for x in data['earlier']['primary_benefits']])
    ledger={'description':'Tables compiled from existing frozen results; no training or new analysis.', 'aggregation':'Continuation BA is unweighted mean of finite condition BA within each task, excluding undefined single-class cells.', 'files':{k:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in files.items()},'primary_rates_verified':True}
    assert abs(effects['cued_-4_change','uncued']['hit_delta']-19.791666666666666/100)<1e-12
    (ROOT/'research/results_sources.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    return out
