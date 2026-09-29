from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "models" / "stage2_mappo"

def run(cmd):
    r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    if r.returncode:
        print(r.stdout); print(r.stderr)
        raise SystemExit(r.returncode)
    return r.stdout

def parse_behavior(text):
    rows=[]
    lines=text.splitlines()
    labels=[
        ("Collision","Collision"),
        ("Completion","Completion"),
        ("Reward","Reward"),
        ("Steps","Steps"),
        ("Near collisions","Near collisions"),
        ("Min distance (m)","Min distance (m)"),
        ("Min TTC (s)","Min TTC (s)"),
    ]
    i=0
    while i<len(lines):
        if "FINAL COMPARISON" not in lines[i]:
            i+=1; continue
        j=i+1
        while j<len(lines) and not lines[j].startswith("Metric"):
            j+=1
        if j>=len(lines): break
        names=lines[j].split()[1:]
        vals={}
        j+=2
        while j<len(lines):
            line=lines[j].strip()
            if not line or line.startswith("ACTION DISTRIBUTION") or line.startswith("="):
                break
            for prefix,key in labels:
                if line.startswith(prefix):
                    raw=line[len(prefix):].split()
                    if len(raw)>=len(names):
                        vals[key]=[float(x) for x in raw[:len(names)]]
                    break
            j+=1
        for k,name in enumerate(names):
            rows.append({
                "Model":name,"Type":"Independent PPO / Stage-1",
                "Collision rate":vals.get("Collision",[None]*len(names))[k],
                "Completion rate":vals.get("Completion",[None]*len(names))[k],
                "Reward":vals.get("Reward",[None]*len(names))[k],
                "Episode length":vals.get("Steps",[None]*len(names))[k],
                "Near collisions":vals.get("Near collisions",[None]*len(names))[k],
                "Min distance (m)":vals.get("Min distance (m)",[None]*len(names))[k],
                "Min TTC (s)":vals.get("Min TTC (s)",[None]*len(names))[k],
            })
        i=j
    return rows

def load_ckpt_eval(background):
    import torch
    sys.path.insert(0,str(ROOT/"src"/"stage2"/"training"))
    from train_mappo import MAPPOConfig, MAPPOTrainer
    ckpt_path=RESULTS/"final.pt"
    if not ckpt_path.exists(): raise FileNotFoundError(ckpt_path)
    cfg=MAPPOConfig(device="auto",background_traffic=background,eval_interval=0,save_interval=0)
    t=MAPPOTrainer(cfg)
    ck=torch.load(ckpt_path,map_location=t.device)
    for aid in t.agent_ids:
        t.actors[aid].load_state_dict(ck["actors"][aid])
        t.observation_rms[aid].load_state_dict(ck["observation_rms"][aid])
    t.critic.load_state_dict(ck["critic"])
    return t.evaluate(episodes=20,background_traffic=background)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-stage1",action="store_true")
    a=ap.parse_args()
    rows=[]
    if not a.skip_stage1:
        print("Running existing Stage-1 PPO comparison...")
        rows += parse_behavior(run([sys.executable,"-m","src.evaluation.compare_behaviors"]))
    print("Evaluating existing Stage-2 MAPPO checkpoint...")
    full=load_ckpt_eval(True)
    nobg=load_ckpt_eval(False)
    for name,r in [("MAPPO (full traffic)",full),("MAPPO (no background)",nobg)]:
        rows.append({
            "Model":name,"Type":"Stage-2 MAPPO",
            "Collision rate":r.get("collision_rate"),
            "Completion rate":r.get("completion_rate"),
            "Reward":r.get("mean_reward"),
            "Episode length":r.get("mean_episode_length"),
            "Near collisions":r.get("mean_near_collisions"),
            "Min distance (m)":r.get("mean_minimum_distance"),
            "Min TTC (s)":r.get("mean_minimum_ttc"),
        })
    preferred=["aggressive_seed45","neutral_seed46","conservative_seed50"]
    chosen=[x for x in rows if x["Model"] in preferred or x["Type"]=="Stage-2 MAPPO"]
    df=pd.DataFrame(chosen)
    out=ROOT/"models"/"comparison"; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"stage2_model_comparison.csv",index=False)
    df.to_json(out/"stage2_model_comparison.json",orient="records",indent=2)
    print("\n"+"="*120)
    print("STAGE 2 MODEL COMPARISON")
    print("="*120)
    print(df.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print(f"\nSaved: {out/'stage2_model_comparison.csv'}")
    print(f"Saved: {out/'stage2_model_comparison.json'}")
if __name__=="__main__":
    main()
