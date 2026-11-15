import ast
import copy
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / 'notebooks/voice-cloning-benchmark.ipynb'


def load_helpers():
    book = json.loads(NOTEBOOK.read_text())
    scope = {}
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'BENCHMARK_CORE_SOURCE','BENCHMARK_RUNNER_SOURCE'} for t in node.targets):
                exec(compile(ast.literal_eval(node.value),'notebook-benchmark-core','exec'),scope)
    return scope


def fixture_data():
    samples = {'run_id':'test-run','count':2,'source_seconds':8.0,'speakers':['Sheldon Cooper'],
               'dataset':'bazinga/bazinga','dataset_revision':'fixed-revision',
               'test_episodes':['TheBigBangTheory.Season01.Episode01'],
               'reference_splits':['train'],'seed':42,'gpu':'test GPU'}
    registry = {'qwen':{'name':'Qwen3-TTS-12Hz-0.6B-Base','repo':'Qwen/Qwen3-TTS-12Hz-0.6B-Base'},
                'higgs':{'name':'Higgs TTS 3 4B','repo':'bosonai/higgs-tts-3-4b'}}
    records = [dict(run_id='test-run',model='qwen',id=0,seconds=4,time=2,rtf=.5,ttfa=None,memory=3),
               dict(run_id='test-run',model='qwen',id=1,seconds=4,time=4,rtf=1,ttfa=None,memory=4),
               dict(run_id='test-run',model='qwen+Sidon',id=0,seconds=4,time=1,rtf=.25,ttfa=None,memory=3)]
    metrics = [dict(r,wer=wer,cer=wer/2,similarity=.5,duration_error=.1)
               for r,wer in zip(records,[.2,.8,.1])]
    return samples,registry,records,metrics


def test_every_model_gets_both_reference_modes():
    h=load_helpers()
    assert 'planned_variants' in h,'Missing all-model A/B planner'
    assert h['planned_variants'](['qwen','cosy2','higgs'])==['qwen','qwen+Sidon','cosy2','cosy2+Sidon','higgs','higgs+Sidon']


def test_pairs_use_common_ids_and_timing_uses_all_synthesized_records():
    h=load_helpers();assert 'build_benchmark_summary' in h,'Missing benchmark aggregator'
    sample,models,records,metrics=fixture_data()
    result=h['build_benchmark_summary'](models,sample,records,metrics,[],[])
    baseline,clean=result['rows'][:2]
    assert baseline['paired_n']==clean['paired_n']==1
    assert baseline['wer']==pytest.approx(.2)
    assert clean['wer']==pytest.approx(.1)
    assert baseline['n_synth']==2 and clean['n_synth']==1
    assert baseline['time_mean']==pytest.approx(3)
    assert baseline['time_total']==pytest.approx(6)
    assert baseline['ttfa'] is None


def test_missing_metrics_stay_missing_and_higgs_is_visible():
    h=load_helpers();assert 'build_benchmark_summary' in h
    sample,models,records,metrics=fixture_data()
    statuses=[{'Модель':'higgs','Статус':'backend несовместим','Причина':'T4'}]
    result=h['build_benchmark_summary'](models,sample,records,metrics,[],statuses)
    assert len(result['rows'])==4
    for row in result['rows'][2:]:
        assert row['n_synth']==0 and row['wer'] is None
        assert 'T4' in row['reason']
    assert result['rows'][0]['wavlm'] is None


def test_stale_and_duplicate_records_cannot_inflate_counts():
    h=load_helpers();assert 'build_benchmark_summary' in h
    sample,models,records,metrics=fixture_data()
    records += [copy.deepcopy(records[0]),dict(records[1],run_id='old-run',id=99)]
    result=h['build_benchmark_summary'](models,sample,records,metrics,[],[])
    assert result['rows'][0]['n_synth']==2


def test_report_is_one_escaped_table_with_full_model_names_and_dataset_volume():
    h=load_helpers();assert 'render_benchmark_html' in h,'Missing HTML renderer'
    sample,models,records,metrics=fixture_data()
    sample['gpu']='GPU <script>alert(1)</script>'
    summary=h['build_benchmark_summary'](models,sample,records,metrics,[],[])
    html=h['render_benchmark_html'](summary)
    assert html.count('<table')==1
    assert 'Qwen3-TTS-12Hz-0.6B-Base' in html and 'Higgs TTS 3 4B' in html
    assert '<script>alert(1)</script>' not in html and '&lt;script&gt;' in html
    assert '8.00' in html and 'bazinga/bazinga' in html
    assert 'ECAPA' in html and 'WavLM' in html and 'Sidon' in html
    assert 'FLORES' in html


def test_process_runner_runs_all_prepared_models_in_both_modes(tmp_path):
    h=load_helpers();assert 'run_tts_comparison' in h,'Missing paired process runner'
    import sys
    worker=tmp_path/'worker.py'
    worker.write_text('''import json,sys,wave
from pathlib import Path
key,path,jobs_file,out=sys.argv[1:]
out=Path(out);out.mkdir(parents=True,exist_ok=True)
print(json.dumps({'event':'ready'}),flush=True)
for j in json.loads(Path(jobs_file).read_text()):
 p=out/f"{j['id']}.wav"
 with wave.open(str(p),'wb') as w:
  w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\\0\\0'*160)
 print(json.dumps({'event':'result','id':j['id'],'file':str(p),'seconds':.01,'time':.001,'rtf':.1,'ttfa':None,'memory':1,'reference':j['ref']}),flush=True)
''')
    jobs=[{'id':i,'ref':'raw'} for i in range(2)]
    (tmp_path/'jobs.json').write_text(json.dumps(jobs))
    (tmp_path/'sidon-jobs.json').write_text(json.dumps([dict(j,ref='clean') for j in jobs]))
    models={'qwen':{'name':'Qwen3-TTS-12Hz-0.6B-Base','repo':'Qwen/model'},'cosy2':{'name':'CosyVoice2-0.5B','repo':'FunAudioLLM/model'}}
    result=h['run_tts_comparison'](models,{k:sys.executable for k in models},{k:'/weights' for k in models},jobs,tmp_path,worker,'run',3.,100.,True,poll_interval=.02)
    assert len(result['results'])==8
    assert {r['model'] for r in result['results']}=={'qwen','qwen+Sidon','cosy2','cosy2+Sidon'}
    for r in result['results']:
        assert r['reference']==('clean' if r['model'].endswith('+Sidon') else 'raw')
        assert r['run_id']=='run'
    assert all(s['Статус']=='завершена' for s in result['statuses'])
    assert result['compute']>=3.


def test_sidon_failure_never_reuses_stale_clean_jobs(tmp_path):
    h=load_helpers();assert 'run_tts_comparison' in h
    import sys
    worker=tmp_path/'worker.py'
    worker.write_text("import json;print(json.dumps({'event':'ready'}),flush=True)")
    jobs=[{'id':0,'ref':'raw'}]
    (tmp_path/'jobs.json').write_text(json.dumps(jobs))
    (tmp_path/'sidon-jobs.json').write_text(json.dumps([{'id':0,'ref':'stale'}]))
    result=h['run_tts_comparison']({'qwen':{'name':'Qwen3-TTS-12Hz-0.6B-Base','repo':'Qwen/model'}},{'qwen':sys.executable},{'qwen':'/weights'},jobs,tmp_path,worker,'run',0.,100.,False,poll_interval=.02)
    assert len(result['statuses'])==2
    assert result['statuses'][0]['Статус']=='неполный результат'
    assert 'Sidon' in result['statuses'][1]['Причина']
    assert result['statuses'][1]['Статус']=='не запущена'


def test_exhausted_budget_skips_all_variants_without_spawning(tmp_path):
    h=load_helpers();assert 'run_tts_comparison' in h
    result=h['run_tts_comparison']({'qwen':{'name':'Qwen3-TTS-12Hz-0.6B-Base','repo':'Qwen/model'}},
        {'qwen':'/must-not-run'},{'qwen':'/weights'},[{'id':0}],tmp_path,tmp_path/'missing-worker','run',10.,10.,True)
    assert result['results']==[] and result['compute']==10.
    assert len(result['statuses'])==2
    assert all(s['Причина']=='Вычислительный лимит' for s in result['statuses'])


def test_startup_timeout_stops_process_without_counting_initialization_as_inference(tmp_path):
    h=load_helpers();assert 'run_tts_comparison' in h
    import sys,time
    worker=tmp_path/'worker.py';worker.write_text('import time;time.sleep(10)')
    jobs=[{'id':0}];(tmp_path/'jobs.json').write_text(json.dumps(jobs))
    begin=time.monotonic()
    result=h['run_tts_comparison']({'qwen':{'name':'Qwen3-TTS-12Hz-0.6B-Base','repo':'Qwen/model'}},
        {'qwen':sys.executable},{'qwen':'/weights'},jobs,tmp_path,worker,'run',7.,100.,False,startup_timeout=.1,poll_interval=.02)
    assert time.monotonic()-begin<3
    assert result['compute']==7.
    assert result['statuses'][0]['Статус']=='таймаут'


def test_disjoint_ids_in_two_measured_modes_do_not_produce_paired_averages():
    h=load_helpers()
    sample,models,records,metrics=fixture_data()
    metrics=[metrics[1],metrics[2]]
    voice=[{'run_id':'test-run','model':'qwen','id':1,'similarity':.9},
           {'run_id':'test-run','model':'qwen+Sidon','id':0,'similarity':.2}]
    summary=h['build_benchmark_summary'](models,sample,records,metrics,voice,[])
    for row in summary['rows'][:2]:
        assert row['paired_n']==0
        assert row['wer'] is None and row['ecapa'] is None
        assert row['wavlm'] is None
