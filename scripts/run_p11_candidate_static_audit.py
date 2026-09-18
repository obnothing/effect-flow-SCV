"""Run Slither locally on P11 error-review candidates with compiler matching."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_p11_error_candidates import audit_label


SOLC_ROOT=Path.home()/".solc-select"/"artifacts"
SOLC={"0.4":SOLC_ROOT/"solc-0.4.25"/"solc-0.4.25","0.5":SOLC_ROOT/"solc-0.5.17"/"solc-0.5.17",
      "0.6":SOLC_ROOT/"solc-0.6.12"/"solc-0.6.12","0.7":SOLC_ROOT/"solc-0.7.6"/"solc-0.7.6",
      "0.8":SOLC_ROOT/"solc-0.8.35"/"solc-0.8.35"}
LABEL_CHECKS={
    "Reentrancy":("reentrancy",),
    "Access Control":("arbitrary-send","suicidal","tx-origin","unprotected-upgrade","protected-vars"),
    "Arithmetic":("divide-before-multiply","incorrect-exp","incorrect-shift","weak-prng"),
    "Unchecked Return Values":("unchecked-lowlevel","unchecked-send","unused-return","unchecked-transfer"),
    "DoS":("calls-loop","msg-value-loop","costly-loop","unbounded-loop"),
    "Time manipulation":("timestamp","weak-prng"),
}


def compiler_for(text):
    match=re.search(r"pragma\s+solidity\s+([^;]+);",text)
    if not match: return None,"missing"
    constraint=match.group(1)
    versions=re.findall(r"0\.[4-8]",constraint)
    if not versions: return None,constraint
    minor=max(int(value.split(".")[1]) for value in versions)
    key=f"0.{minor}"
    return SOLC.get(key),constraint


def relevant_checks(label,checks):
    prefixes=LABEL_CHECKS[label]
    return [check for check in checks if any(prefix in check for prefix in prefixes)]


def run_slither(source,compiler,json_path):
    command=["slither",str(source),"--solc",str(compiler),"--exclude-dependencies","--json",str(json_path)]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    except subprocess.TimeoutExpired:
        return {"status":"timeout","checks":[],"stderr":"timeout"}
    checks=[]
    if json_path.exists():
        try:
            payload=json.loads(json_path.read_text(encoding="utf-8"))
            checks=sorted({item.get("check","") for item in payload.get("results",{}).get("detectors",[])})
        except json.JSONDecodeError: pass
    return {"status":"completed" if checks or result.returncode==0 else "compile_or_tool_error",
            "returncode":result.returncode,"checks":checks,"stderr":result.stderr[-2000:]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidates",default="results/light_label/p11_error_diagnosis/label_review_candidates.csv")
    parser.add_argument("--raw-root",default="DIVE_Raw_Data/Raw")
    parser.add_argument("--output-dir",default="results/light_label/p11_error_diagnosis/static_tool_audit")
    args=parser.parse_args(); candidates=list(csv.DictReader(Path(args.candidates).open(encoding="utf-8")))
    raw=Path(args.raw_root); source_dir=raw/"PRE"/"Source codes"; out=Path(args.output_dir); raw_json=out/"slither_json"; raw_json.mkdir(parents=True,exist_ok=True)
    tool_availability={name:shutil.which(command) for name,command in {"Slither":"slither","SmartCheck":"smartcheck","Oyente":"oyente","Securify":"securify"}.items()}
    cache={}; rows=[]
    for candidate in candidates:
        row=dict(candidate); identifier=row["id"]
        if not identifier.startswith("dive:"):
            row.update({"audit_status":"source_unavailable","slither_status":"not_run","slither_checks":"[]","slither_relevant_checks":"[]"}); rows.append(row); continue
        contract_id=int(identifier.split(":",1)[1]); source=source_dir/f"{contract_id}.sol"
        if not source.exists():
            row.update({"audit_status":"source_missing","slither_status":"not_run","slither_checks":"[]","slither_relevant_checks":"[]"}); rows.append(row); continue
        if contract_id not in cache:
            text=source.read_text(encoding="utf-8",errors="replace"); compiler,constraint=compiler_for(text)
            if compiler is None or not compiler.exists(): static={"status":"compiler_unavailable","checks":[],"constraint":constraint}
            else: static=run_slither(source,compiler,raw_json/f"{contract_id}.json"); static["constraint"]=constraint; static["compiler"]=str(compiler)
            static["heuristic"]=audit_label(row["label"],text); cache[contract_id]=static
        static=cache[contract_id]; relevant=relevant_checks(row["label"],static["checks"])
        truth=int(row["truth"])
        if static["status"]=="completed" and relevant:
            audit_status="slither_supports_positive" if truth else "slither_supports_model_positive"
        elif static["status"]=="completed":
            audit_status="slither_no_label_specific_finding"
        else: audit_status=static["status"]
        row.update({"source_path":str(source),"compiler_constraint":static.get("constraint",""),"compiler":static.get("compiler",""),
                    "slither_status":static["status"],"slither_checks":json.dumps(static["checks"]),
                    "slither_relevant_checks":json.dumps(relevant),"audit_status":audit_status,
                    "heuristic_signals":"|".join(static["heuristic"]["signals"]),"heuristic_evidence":json.dumps(static["heuristic"]["evidence_lines"]),
                    "slither_stderr":static.get("stderr","")})
        rows.append(row)
    fields=sorted({key for row in rows for key in row}); csv_path=out/"candidate_static_audit.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary={"candidate_count":len(rows),"tool_availability":tool_availability,
             "slither_completed":sum(row.get("slither_status")=="completed" for row in rows),
             "status_counts":{key:sum(row.get("audit_status")==key for row in rows) for key in sorted({row.get("audit_status") for row in rows})},
             "scope":"Slither findings and source heuristics are cross-tool evidence, not ground-truth label adjudication."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
