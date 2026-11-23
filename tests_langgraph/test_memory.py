import json, os, sqlite3, subprocess, sys, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from robot_controller.memory import ObservationStore, canonical, task_id, persist_decision
from robot_controller.schemas import Task, Limits
from robot_controller.telemetry import Ledger
from robot_controller.orchestrator import Controller

TASK = Task('Inspect Station 1.', 'station_1', ('station_1','station_2'))
WORLD = 'a'*64
REC = {'proposal':{'kind':'navigate','target':'station_1'},'executed':'advanced_towards_target',
       'dispatched':True,'pose_before':[0,0,0],'pose_after':[0.5,0,0], 'displacement_m':0.5,
       'arrived_at_target':False,'at_station_1_after':False}

def source(event='event:1', **kw):
    d=dict(world_id=WORLD,task=TASK,episode_id='episode_1',controller_run_id='controller_1',
           source_event_id=event,step=0,writer_id='runner',frame_sha256='b'*64)
    d.update(kw); return d

class Model:
    model='deterministic-test-double'
    def __init__(self): self.calls=[]; self.supervisor_calls=0
    def complete(self, role, system, payload, schema, frame):
        self.calls.append((role,system,payload))
        if role!='supervisor': return {'summary':'The measurement indicates earlier forward movement.'}
        self.supervisor_calls+=1
        if self.supervisor_calls==1:
            return {'decision':'delegate','jobs':[{'role':x,'question':'Assess the recorded measurement.'} for x in ('observe','plan','check')], 'action':None,'reason':'Obtain scoped evidence.'}
        return {'decision':'finish','jobs':[],'action':{'kind':'stop','target':None},'reason':'No motion requested by this software test.'}

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.path=self.root/'memory.sqlite3'
        self.store=ObservationStore(self.path,'neutral-test')
    def tearDown(self): self.tmp.cleanup()
    def test_measured_write_and_recall(self):
        receipt=self.store.append_execution(REC,**source())
        snap=self.store.snapshot(world_id=WORLD,task=TASK).payload()
        self.assertEqual(snap['record_ids'],[receipt['id']]); self.assertEqual(snap['records'][0]['payload']['displacement_m'],.5)
    def test_new_process_reopens_store_and_reads_exact_id(self):
        receipt=self.store.append_execution(REC,**source())
        script="from robot_controller.memory import ObservationStore; from robot_controller.schemas import Task; import json,sys; s=ObservationStore(sys.argv[1],'neutral-test'); t=Task('Inspect Station 1.','station_1',('station_1','station_2')); print(json.dumps(s.snapshot(world_id='a'*64,task=t).payload()))"
        p=subprocess.run([sys.executable,'-c',script,str(self.path)],capture_output=True,text=True,check=True)
        snap=json.loads(p.stdout)
        self.assertEqual(snap['store_id'],self.store.store_id); self.assertEqual(snap['record_ids'],[receipt['id']])
    def test_idempotent_write_and_conflict(self):
        a=self.store.append_execution(REC,**source()); b=self.store.append_execution(REC,**source())
        self.assertTrue(a['created']); self.assertFalse(b['created'])
        with self.assertRaises(ValueError): self.store.append_execution({**REC,'displacement_m':.6},**source())
    def test_scope_isolation(self):
        self.store.append_execution(REC,**source())
        self.assertEqual(self.store.snapshot(world_id='c'*64,task=TASK).payload()['records'],[])
        self.assertEqual(ObservationStore(self.path,'other').snapshot(world_id=WORLD,task=TASK).payload()['records'],[])
        other=Task('Inspect Station 2.','station_2',('station_1','station_2'))
        self.assertEqual(self.store.snapshot(world_id=WORLD,task=other).payload()['records'],[])
    def test_claims_persist_but_never_automatically_enter_policy_snapshot(self):
        self.store.append_claim('A blue crate is visible.','observe',**source(writer_id='worker_0_0'))
        self.assertEqual(len(ObservationStore(self.path,'neutral-test').archive('model_claim')),1)
        self.assertEqual(self.store.snapshot(world_id=WORLD,task=TASK).payload()['records'],[])
    def test_free_text_not_promoted_into_execution_fact(self):
        self.store.append_execution({**REC,'report':'A blue crate is visible.','reason':'A descriptive claim.'},**source())
        self.assertNotIn('crate',canonical(self.store.snapshot(world_id=WORLD,task=TASK).payload()))
    def test_all_recipients_get_exact_snapshot_and_read_events(self):
        receipt=self.store.append_execution(REC,**source())
        snapshot=self.store.snapshot(world_id=WORLD,task=TASK)
        model=Model(); ledger=Ledger(self.root/'logs')
        result=Controller(TASK,model,ledger,limits=Limits(max_rounds=2),memory_snapshot=snapshot).run()
        self.assertEqual(result['workers_completed'],3)
        self.assertEqual(len(model.calls),5)
        reads=[e for e in ledger.events if e['event_type']=='memory_read']
        self.assertEqual(len(reads),5)
        self.assertEqual(set(e['agent_id'] for e in reads),{'supervisor','worker_0_0','worker_0_1','worker_0_2'})
        for role,system,p in model.calls:
            self.assertEqual(p['persistent_task_memory']['record_ids'],[receipt['id']])
            self.assertEqual(p['operator_task'],TASK.payload())
        for e in ledger.events:
            if e['event_type']=='model_input': self.assertIn(e['parent_event_ids'][0],[x['event_id'] for x in reads])
    def test_memory_off_preserves_existing_controller_contract(self):
        model=Model(); ledger=Ledger(self.root/'logs')
        Controller(TASK,model,ledger,limits=Limits(max_rounds=2)).run()
        self.assertFalse(any(e['event_type']=='memory_read' for e in ledger.events))
        self.assertTrue(all('persistent_task_memory' not in p for _,_,p in model.calls))
    def test_wrong_task_snapshot_rejected(self):
        other=Task('Inspect Station 2.','station_2',('station_1','station_2'))
        with self.assertRaises(ValueError): Controller(other,Model(),Ledger(self.root/'logs'),memory_snapshot=self.store.snapshot(world_id=WORLD,task=TASK))
    def test_read_limit_and_durable_audit(self):
        for i in range(7): self.store.append_execution(REC,**source('event:'+str(i)))
        snap=self.store.snapshot(world_id=WORLD,task=TASK,limit=3).payload()
        self.assertEqual(len(snap['records']),3)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE operation='write'").fetchone()[0],7)
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE operation='snapshot_read'").fetchone()[0],1)
    def test_nonfinite_measurement_rejected(self):
        with self.assertRaises(ValueError): self.store.append_execution({**REC,'pose_after':[float('nan'),0,0]},**source())
        self.assertEqual(self.store.archive(),[])
    def test_hash_corruption_detected(self):
        self.store.append_execution(REC,**source())
        with self.store.connection() as db: db.execute("UPDATE records SET body='{}'")
        with self.assertRaises(ValueError): self.store.snapshot(world_id=WORLD,task=TASK)
    def test_concurrent_writes_no_loss(self):
        def write(i): return self.store.append_execution(REC,**source('event:'+str(i)))
        with ThreadPoolExecutor(max_workers=4) as pool: receipts=list(pool.map(write,range(12)))
        self.assertEqual(len(self.store.archive()),12); self.assertEqual(len(set(r['id'] for r in receipts)),12)
    def test_decision_persists_executor_fact_and_separate_worker_claims(self):
        ledger=Ledger(self.root/'logs'); model=Model()
        Controller(TASK,model,ledger,limits=Limits(max_rounds=2)).run()
        eid=ledger.emit('executor_result','executor',**REC)
        writes=persist_decision(self.store,ledger,TASK,WORLD,'episode_1',0,REC,eid,'b'*64)
        self.assertEqual(len(writes),6)
        self.assertEqual(len(self.store.archive('execution_fact')),1)
        self.assertEqual(len(self.store.archive('model_claim')),5)
        self.assertTrue(all(r['frame_sha256'] is None for r in self.store.archive('model_claim')))
        self.assertEqual(len(self.store.snapshot(world_id=WORLD,task=TASK).payload()['records']),1)

if __name__=='__main__': unittest.main()
