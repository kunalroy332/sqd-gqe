# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json,glob,os,collections,numpy as np,sys
import pathlib
REPO=pathlib.Path(__file__).resolve().parents[3]
ROOT=str(REPO/'circuit_analysis')
INF=os.environ.get('INF_DIR','/tmp/inf')
DMRG={"sno32":-479.193381728830,"h2o_s19":-555.108832785690,"2h2o_s22":-631.083312007953,"h2o_m_no23":-555.125158274414,"llno28":-479.315056933618,"h2o_l28":-555.135057773478,"2h2o_m_no29":-631.118615946406,"h2o_ll33":-555.137396193158,"2h2o_l35":-631.132237762872,"L46":-479.314216460012,"sno38":-479.302278277561}
NAME={"sno32":"SnOH3-me S 32q","sno38":"SnOH3-me M 38q","L46":"SnOH3-me L 46q","llno28":"SnOH3-me LL 56q","h2o_s19":"h2o S 38q","h2o_m_no23":"h2o M 46q","h2o_l28":"h2o L 56q","h2o_ll33":"h2o LL 66q","2h2o_s22":"2h2o S 44q","2h2o_m_no29":"2h2o M 58q","2h2o_l35":"2h2o L 70q"}
MK=sorted(DMRG,key=len,reverse=True)
def split(tag):
    m=next(k for k in MK if tag.startswith(k+"_")); return m,tag[len(m)+1:].rsplit("_",1)[0]
SETS={"T1":(ROOT+"/rerecovery",ROOT+"/data",ROOT+"/chem"),"HF":(INF+"/inference_hf/circuit_analysis_rerecovery",INF+"/inference_hf/circuit_analysis",None),"tuned":(INF+"/inference_tuned/circuit_analysis_rerecovery",INF+"/inference_tuned/circuit_analysis",None)}
R=collections.defaultdict(dict)   # (mol,lin) -> set -> info
for s,(rd,dd,cd) in SETS.items():
    rc=collections.defaultdict(dict)
    for f in glob.glob(rd+"/*__*.json"):
        tag,tg=os.path.basename(f)[:-5].split("__"); j=json.load(open(f))
        if "final_energy" in j and tag.endswith("_trained"): rc[tag][tg]=j
    for tag,d in rc.items():
        if not {"best","second"}<=set(d): continue
        m,lin=split(tag); 
        try: run=json.load(open(f"{dd}/{tag}.json"))
        except Exception: continue
        circ={c["index"]:c for c in run["circuits"]}
        def info(tg):
            j=d[tg]; idx=j["circuit_indices"][0]; c=circ.get(idx,{})
            es=c.get("exc_stats") or {}
            return {"E":j["final_energy"],"err":(j["final_energy"]-DMRG[m])*1000,"depth":c.get("depth_live"),"valid":es.get("frac_valid"),"hf":es.get("valid_hf"),"hl":es.get("valid_homo_lumo_any"),"a":es.get("valid_alpha_only"),"b":es.get("valid_beta_only"),"both":es.get("valid_alpha_and_beta")}
        R[(m,lin)][s]={"best":info("best"),"second":info("second"),"merged_err":(d["merged"]["final_energy"]-DMRG[m])*1000 if "merged" in d else None}
        # shot-based decomposition for Training 1 from the backfill
        if s=="T1":
            try:
                ch=json.load(open(f"{cd}/{m}_{lin}_trained.json"))["circuits"]; ci={c["index"]:c for c in ch}
                for tg in ("best","second"):
                    idx=d[tg]["circuit_indices"][0]; e=ci[idx]["exc_stats"]
                    R[(m,lin)][s][tg].update({"valid":e["frac_valid"],"hf":e["valid_hf"],"hl":e["valid_homo_lumo_any"],"a":e["valid_alpha_only"],"b":e["valid_beta_only"],"both":e["valid_alpha_and_beta"]})
            except Exception: pass
import pickle; pickle.dump(dict(R),open(INF+'/R3.pkl','wb'))
print("run cells with at least one set:",len(R)); 
cnt=collections.Counter(tuple(sorted(v)) for v in R.values()); print(dict(cnt))
