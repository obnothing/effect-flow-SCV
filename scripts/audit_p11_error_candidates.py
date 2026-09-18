"""Conservative source-level audit for P11 label-review candidates.

This script does not claim to replace Slither, Oyente, SmartCheck, or Securify.
It records tool availability and source-level evidence for manual review.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
LABELS=["Reentrancy","Access Control","Arithmetic","Unchecked Return Values","DoS","Time manipulation"]


def source_lines(text, pattern, limit=3):
    rows=[]
    for number,line in enumerate(text.splitlines(),1):
        if re.search(pattern,line,re.I): rows.append(f"{number}:{line.strip()[:300]}")
        if len(rows)>=limit: break
    return rows


def pragma_version(text):
    match=re.search(r"pragma\s+solidity\s+([^;]+);",text)
    return match.group(1).strip() if match else "missing"


def audit_label(label,text):
    lowered=text.lower(); pragma=pragma_version(text)
    calls=source_lines(text,r"\.call(?:\{|\.value)?|\.delegatecall|\.send\(|\.transfer\(")
    loops=source_lines(text,r"\bfor\s*\(|\bwhile\s*\(|\bdo\s*\{")
    time=source_lines(text,r"block\.timestamp|\bnow\b|block\.number")
    arithmetic=source_lines(text,r"(?<![=!<>])\+(?![+=])|(?<![=!<>])-|\*|/(?!/)|%")
    auth=source_lines(text,r"onlyowner|onlyadmin|require\s*\([^\n;]*msg\.sender|msg\.sender\s*==|tx\.origin")
    safe_math=bool(re.search(r"safemath|using\s+safemath",lowered))
    non_reentrant=bool(re.search(r"nonreentrant|reentrancyguard",lowered))
    records=[]
    if label=="Reentrancy":
        if calls:
            records.append(("potential_external_call_surface",calls))
        if non_reentrant: records.append(("reentrancy_guard_present",source_lines(text,r"nonreentrant|reentrancyguard")))
        if not calls: records.append(("no_direct_external_call_surface",[]))
    elif label=="Access Control":
        if re.search(r"tx\.origin",lowered): records.append(("tx_origin_authorization_pattern",source_lines(text,r"tx\.origin")))
        if auth: records.append(("authorization_logic_present",auth))
        if not auth: records.append(("no_simple_authorization_pattern",[]))
    elif label=="Arithmetic":
        legacy=bool(re.search(r"(?:\^|~|>=|>)?\s*0\.[0-7](?:\.|$)",pragma))
        if arithmetic: records.append(("arithmetic_operations_present",arithmetic))
        if legacy and not safe_math: records.append(("legacy_compiler_without_detected_safemath",[]))
        if safe_math: records.append(("safemath_pattern_present",source_lines(text,r"safemath|using\s+safemath")))
    elif label=="Unchecked Return Values":
        unchecked=[]
        for number,line in enumerate(text.splitlines(),1):
            if re.search(r"\.call(?:\{|\.value)?|\.delegatecall|\.send\(",line,re.I):
                checked=bool(re.search(r"require\s*\(|assert\s*\(|if\s*\(|success|\bok\b|\bresult\b",line,re.I))
                unchecked.append(f"{number}:{'checked_or_assigned' if checked else 'potentially_unchecked'}:{line.strip()[:300]}")
        if unchecked: records.append(("low_level_call_review",unchecked[:3]))
        else: records.append(("no_direct_low_level_call_surface",[]))
    elif label=="DoS":
        if loops: records.append(("loop_present",loops))
        if loops and calls: records.append(("loop_and_external_call_surface",loops[:2]+calls[:2]))
        if not loops: records.append(("no_direct_loop_surface",[]))
    elif label=="Time manipulation":
        if time: records.append(("timestamp_or_block_number_present",time))
        else: records.append(("no_timestamp_or_block_number_pattern",[]))
    return {"pragma":pragma,"signals":[name for name,_ in records],
            "evidence_lines":[line for _,lines in records for line in lines],
            "external_call_count":len(calls),"loop_count":len(loops),
            "time_pattern_count":len(time),"safe_math_detected":safe_math,
            "non_reentrant_detected":non_reentrant}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidates",default="results/light_label/p11_error_diagnosis/label_review_candidates.csv")
    parser.add_argument("--raw-root",required=True)
    parser.add_argument("--output",default="results/light_label/p11_error_diagnosis/source_audit.csv")
    args=parser.parse_args(); raw=Path(args.raw_root); candidates=list(csv.DictReader(Path(args.candidates).open(encoding="utf-8")))
    source_dir=raw/"PRE"/"Source codes"; addresses=[line.strip() for line in (raw/"DIVE_Samples.csv").read_text(encoding="utf-8").splitlines()[1:]]
    tools={name:shutil.which(command) for name,command in {"Slither":"slither","SmartCheck":"smartcheck","Oyente":"oyente","Securify":"securify"}.items()}
    rows=[]
    for candidate in candidates:
        item=dict(candidate); identifier=item["id"]
        if identifier.startswith("dive:"):
            contract_id=int(identifier.split(":",1)[1]); source=source_dir/f"{contract_id}.sol"
            item["source_path"]=str(source); item["contract_address"]=addresses[contract_id-1] if contract_id<=len(addresses) else ""
            item["source_available"]=source.exists()
            if source.exists():
                text=source.read_text(encoding="utf-8",errors="replace")
                item["source_sha256"]=hashlib.sha256(text.encode()).hexdigest()
                audit=audit_label(item["label"],text)
                item.update({"pragma":audit["pragma"],"heuristic_signals":"|".join(audit["signals"]),
                             "evidence_lines":json.dumps(audit["evidence_lines"]),"external_call_count":audit["external_call_count"],
                             "loop_count":audit["loop_count"],"time_pattern_count":audit["time_pattern_count"],
                             "safe_math_detected":audit["safe_math_detected"],"non_reentrant_detected":audit["non_reentrant_detected"],
                             "audit_status":"source_signal_only"})
            else: item["audit_status"]="source_missing"
        else:
            item.update({"source_path":"","contract_address":identifier.split(":",1)[1] if ":" in identifier else "",
                         "source_available":False,"audit_status":"source_not_available_in_DIVE_raw"})
        rows.append(item)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for row in rows for key in row})
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary={"candidate_count":len(rows),"dive_source_available":sum(bool(row.get("source_available")) for row in rows),
             "tool_availability":tools,"audit_scope":"source-level heuristic evidence only; no tool result is inferred when unavailable"}
    output.with_suffix(".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
