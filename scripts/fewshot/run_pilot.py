"""Local Pilot-0 orchestration: strict LOVO, fixed support episodes, M0-M5."""

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import LightLabelDataset,collate_light_label,collate_light_label_inference
from polarity_query_model import build_model
from fewshot.train_lovo_base import train_one as train_base
from fewshot.pilot_core import (gradient_residual_basis,initialize_random_query,novel_forward,novel_loss,
    query_mixture,random_orthogonal_basis,strict_metric)

CONFIG_PATH=ROOT/"configs/fewshot_query_pilot/pilot0.yaml"
METHODS=("M0_random_query","M1_best_old_query","M2_old_query_mixture","M3_mixture_random_residual","M4_mixture_gradient_residual","M5_full_finetuning")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


class OpcodeOnlyDataset(Dataset):
    def __init__(self,path,max_len):
        data=torch.load(path,map_location="cpu",weights_only=False)
        if "labels" in data: raise ValueError("valid cache must defer labels until final evaluation")
        self.token_ids,self.offsets,self.ids=data["token_ids"],data["offsets"],data["ids"]
        self.original_lengths=data["original_lengths"];self.max_len=int(max_len)
    def __len__(self):return len(self.ids)
    def sequence_length(self,i):return min(int(self.offsets[i+1]-self.offsets[i]),self.max_len)
    def __getitem__(self,i):
        left,right=int(self.offsets[i]),int(self.offsets[i+1]);v=self.token_ids[left:min(right,left+self.max_len)].long()
        return {"id":self.ids[i],"input_ids":v,"length":len(v),"original_length":int(self.original_lengths[i])}


def read_valid_labels(path):
    """This is called only after every checkpoint and support fit is fixed."""
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line);rows.append((str(row["id"]),list(map(int,row["multi_labels"]))))
    return rows


def read_split_manifest(label):
    path=ROOT/"results/fewshot_query_pilot"/"splits"/f"{label.replace(' ','_')}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def collate_hidden(records,indices,device):
    selected=[records[i] for i in indices]
    maxlen=max(item["hidden"].shape[0] for item in selected);dim=selected[0]["hidden"].shape[-1]
    hidden=torch.zeros((len(selected),maxlen,dim),dtype=torch.float16)
    mask=torch.zeros((len(selected),maxlen),dtype=torch.bool)
    labels=torch.tensor([item["label"] for item in selected],dtype=torch.float32)
    ids=[]
    for i,item in enumerate(selected):
        n=item["hidden"].shape[0];hidden[i,:n]=item["hidden"];mask[i,:n]=True;ids.append(item["id"])
    return hidden.to(device),mask.to(device),labels.to(device),ids


@torch.no_grad()
def encode_support(base,train_data,id_map,ids,tokenizer,novel_index,batch_size=2):
    base.eval();device=torch.device("cuda");items=[train_data[id_map[cid]] for cid in ids];result=[]
    for start in range(0,len(items),batch_size):
        batch=collate_light_label(items[start:start+batch_size],tokenizer.pad_token_id)
        x=batch["input_ids"].to(device);mask=batch["mask"].to(device)
        with torch.autocast("cuda",dtype=torch.float16):hidden=base.encode_tokens(x,batch["lengths"],mask)
        for j,n in enumerate(batch["lengths"].tolist()):
            result.append({"id":batch["ids"][j],"hidden":hidden[j,:n].detach().half().cpu(),"label":float(batch["labels"][j,novel_index])})
    return result


def matched_scorer(base,seed):
    from fewshot.pilot_core import match_old_distribution
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    scorer=match_old_distribution(torch.randn(base.label_scorer.shape[-1],generator=gen),base.label_scorer.detach().cpu())
    biases=[]
    for p in range(2):
        ref=base.branch_bias.detach().cpu()[:,p]
        r=torch.randn(1,generator=gen)
        biases.append(float(r*ref.std(unbiased=False).clamp_min(1e-6)+ref.mean()))
    return torch.nn.Parameter(scorer.cuda()),torch.nn.Parameter(torch.tensor(biases,dtype=torch.float32,device="cuda"))


def plain_scorer_and_bias(base,seed):
    w,b=matched_scorer(base,seed)
    return w.detach().clone(),b.detach().clone()


def support_query_gradients(base,records,qmix,scorer,bias,aux_weight,polarity):
    gradients=[]
    for record in records:
        h,mask,y,_=collate_hidden([record],[0],"cuda")
        qp=qmix[0].detach().clone().requires_grad_(True);qm=qmix[1].detach().clone().requires_grad_(True)
        with torch.autocast("cuda",dtype=torch.float16):
            out=novel_forward(base,h,mask,torch.stack([qp,qm]),scorer,bias)
            loss,_,_=novel_loss(out,y,aux_weight)
        gradients.append(torch.autograd.grad(loss,(qp,qm))[polarity].detach().float().cpu())
    return torch.stack(gradients)


def model_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter(value):
    return torch.nn.Parameter(value.detach().clone().to("cuda"))


def fit_head(base,records,labels,query_fn,parameters,config,seed,method):
    optimizer=torch.optim.AdamW(parameters,lr=float(config["adapt_learning_rate"]),weight_decay=float(config["adapt_weight_decay"]))
    history=[];rng=np.random.default_rng(int(seed))
    for epoch in range(int(config["adapt_epochs"])):
        order=rng.permutation(len(records));losses=[];base.eval()
        for offset in range(0,len(order),int(config["adapt_batch_size"])):
            idx=order[offset:offset+int(config["adapt_batch_size"])].tolist()
            h,mask,y,_=collate_hidden(records,idx,"cuda");optimizer.zero_grad(set_to_none=True)
            queries,scorer,bias=query_fn()
            with torch.autocast("cuda",dtype=torch.float16):
                output=novel_forward(base,h,mask,queries,scorer,bias);loss,_,_=novel_loss(output,y,float(config["auxiliary_weight"]))
            if not torch.isfinite(loss):raise RuntimeError(f"nonfinite {method} support loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step();losses.append(float(loss.detach()))
        row={"epoch":epoch+1,"support_loss":float(np.mean(losses))};history.append(row)
        print(f"[adapt:{method}] epoch={epoch+1}/{config['adapt_epochs']} support_loss={row['support_loss']:.6f}",flush=True)
    return history


def to_cpu_head(method,queries,scorer,bias,metadata=None):
    return {"method":method,"queries":queries.detach().float().cpu(),"scorer":scorer.detach().float().cpu(),
            "bias":bias.detach().float().cpu(),"metadata":metadata or {}}


def train_m0_m2(base,records,labels,novel_index,old_names,config,seed):
    heads={};curves={};init_seed=42+novel_index*1000+seed
    random_q,qstats=initialize_random_query(base.queries.detach(),init_seed+1)
    scorer0,bias0=matched_scorer(base,init_seed+2)
    q0=torch.nn.Parameter(random_q.cuda())
    build0=lambda:(q0,scorer0,bias0)
    curves["M0_random_query"]=fit_head(base,records,labels,build0,[q0,scorer0,bias0],config,init_seed+3,"M0")
    heads["M0_random_query"]=to_cpu_head("M0_random_query",q0,scorer0,bias0,{"random_query_stats":qstats})

    source_scores=[];source_heads=[];old_names_for_query=old_names
    for source_index,source_name in enumerate(old_names_for_query):
        q=base.queries[source_index].detach().clone()
        w,b=matched_scorer(base,init_seed+2)
        build=lambda q=q,w=w,b=b:(q,w,b)
        curve=fit_head(base,records,labels,build,[w,b],config,init_seed+100+source_index,"M1_"+source_name)
        source_loss=curve[-1]["support_loss"]
        source_scores.append({"source_label":source_name,"support_loss":source_loss})
        source_heads.append(to_cpu_head("M1_best_old_query",q,w,b,{"source_label":source_name,"support_loss":source_loss}))
    best_idx=min(range(len(source_scores)),key=lambda i:source_scores[i]["support_loss"])
    heads["M1_best_old_query"]=source_heads[best_idx];heads["M1_best_old_query"]["metadata"]["all_source_support_losses"]=source_scores
    curves["M1_best_old_query"]=[{"epoch":i+1,"support_loss":source_scores[best_idx]["support_loss"]} for i in range(int(config["adapt_epochs"]))]

    alpha_p=torch.nn.Parameter(torch.zeros(len(old_names),device="cuda"));alpha_m=torch.nn.Parameter(torch.zeros(len(old_names),device="cuda"))
    scorer2,bias2=matched_scorer(base,init_seed+2)
    build2=lambda:(query_mixture(base.queries,alpha_p,alpha_m),scorer2,bias2)
    curves["M2_old_query_mixture"]=fit_head(base,records,labels,build2,[alpha_p,alpha_m,scorer2,bias2],config,init_seed+4,"M2")
    qmix=query_mixture(base.queries,alpha_p,alpha_m)
    weights={"alpha_plus":torch.softmax(alpha_p.detach(),0).cpu().tolist(),"alpha_minus":torch.softmax(alpha_m.detach(),0).cpu().tolist(),"old_label_order":old_names}
    heads["M2_old_query_mixture"]=to_cpu_head("M2_old_query_mixture",qmix,scorer2,bias2,{"mixture":weights,"source_support_losses":source_scores})
    return heads,curves,weights,{"random_query_stats":qstats,"source_scores":source_scores,"m2_alpha_params":len(alpha_p)+len(alpha_m)}


def support_gradient_matrix(base,records,qmix,scorer,bias,aux_weight):
    plus_rows=[];minus_rows=[];plus_sq=minus_sq=res_plus_sq=res_minus_sq=0.0
    for record in records:
        h,mask,y,_=collate_hidden([record],[0],"cuda")
        qplus=qmix[0].detach().clone().requires_grad_(True)
        qminus=qmix[1].detach().clone().requires_grad_(True)
        with torch.autocast("cuda",dtype=torch.float16):
            output=novel_forward(base,h,mask,torch.stack([qplus,qminus]),scorer,bias)
            loss,_,_=novel_loss(output,y,aux_weight)
        gplus,gminus=torch.autograd.grad(loss,(qplus,qminus),retain_graph=False,create_graph=False)
        plus_rows.append(gplus.detach().float().cpu());minus_rows.append(gminus.detach().float().cpu())
        plus_sq+=float(gplus.float().square().sum());minus_sq+=float(gminus.float().square().sum())
    qplus_span=base.queries.detach().cpu()[:,0,:]
    qminus_span=base.queries.detach().cpu()[:,1,:]
    bplus,dplus=gradient_residual_basis(qplus_span,torch.stack(plus_rows),4)
    bminus,dminus=gradient_residual_basis(qminus_span,torch.stack(minus_rows),4)
    diagnostics={"R_pos":dplus["ratio"],"R_neg":dminus["ratio"],"C4_pos":dplus["c4"],"C4_neg":dminus["c4"],
        "rank_pos":dplus["rank"],"rank_neg":dminus["rank"],"effective_rank_pos":dplus.get("effective_rank",0),
        "effective_rank_neg":dminus.get("effective_rank",0),"orthogonality_pos":dplus["orthogonality"],
        "orthogonality_neg":dminus["orthogonality"]}
    return bplus.to("cuda"),bminus.to("cuda"),diagnostics


def train_residual(base,records,labels,config,seed,mix_head,method,bplus,bminus):
    qmix=mix_head["queries"].to("cuda")
    scorer=mix_head["scorer"].to("cuda");bias=mix_head["bias"].to("cuda")
    rplus,rminus=bplus.shape[1],bminus.shape[1]
    cplus=torch.nn.Parameter(torch.zeros(rplus,device="cuda"));cminus=torch.nn.Parameter(torch.zeros(rminus,device="cuda"))
    params=[p for p in (cplus,cminus) if p.numel()]
    def query_fn():
        plus=qmix[0]+(bplus@cplus if rplus else 0)
        minus=qmix[1]+(bminus@cminus if rminus else 0)
        return torch.stack([plus,minus]),scorer,bias
    curve=fit_head(base,records,labels,query_fn,params,config,seed,method) if params else []
    queries,_,_=query_fn()
    return to_cpu_head(method,queries,scorer,bias,{"residual_rank":max(rplus,rminus),"rank_pos":rplus,"rank_neg":rminus,
        "coeff_pos":cplus.detach().cpu(),"coeff_neg":cminus.detach().cpu()}),curve


def fit_m5(base_model,train_data,id_map,ids,tokenizer,novel_index,config,seed,initial_query,initial_scorer,initial_bias):
    import copy
    model=copy.deepcopy(base_model).train()
    for name,param in model.named_parameters():
        param.requires_grad=name.startswith(("embedding.","encoder.","cross_attention."))
    query=torch.nn.Parameter(initial_query.detach().clone().to("cuda"))
    scorer=torch.nn.Parameter(initial_scorer.detach().clone().to("cuda"))
    bias=torch.nn.Parameter(initial_bias.detach().clone().to("cuda"))
    params=[p for p in model.parameters() if p.requires_grad]+[query,scorer,bias]
    optimizer=torch.optim.AdamW(params,lr=float(config["full_ft_learning_rate"]),weight_decay=float(config["full_ft_weight_decay"]))
    indices=[id_map[cid] for cid in ids]
    data=LightLabelDataset(ROOT/config["cache_dir"]/f"train_max{config['max_len']}.pt",runtime_max_len=config["max_len"],indices=indices)
    loader=DataLoader(data,batch_size=int(config["full_ft_batch_size"]),shuffle=True,num_workers=0,
        collate_fn=partial(collate_light_label,pad_id=tokenizer.pad_token_id),generator=torch.Generator().manual_seed(seed))
    curves=[];amp=True
    for epoch in range(int(config["full_ft_epochs"])):
        model.train();losses=[]
        for batch in loader:
            x=batch["input_ids"].cuda(non_blocking=True);mask=batch["mask"].cuda(non_blocking=True)
            y=batch["labels"][:,novel_index].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                hidden=model.encode_tokens(x,batch["lengths"],mask)
                output=novel_forward(model,hidden,mask,torch.stack([query[0],query[1]]),scorer,bias)
                loss,_,_=novel_loss(output,y,float(config["auxiliary_weight"]))
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();losses.append(float(loss.detach()))
        row={"epoch":epoch+1,"support_loss":float(np.mean(losses))};curves.append(row)
        print(f"[adapt:M5] epoch={epoch+1}/{config['full_ft_epochs']} support_loss={row['support_loss']:.6f}",flush=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    head=to_cpu_head("M5_full_finetuning",torch.stack([query[0],query[1]]),scorer,bias,
        {"fine_tuned_state":state,"trainable_encoder_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
         "trainable_scorer_params":scorer.numel()+bias.numel(),"trainable_query_params":query.numel()})
    return head,curves


def encode_episode(base,train_data,id_map,episode,tokenizer,novel_index):
    ids=episode["positive_ids"]+episode["negative_ids"]
    records=encode_support(base,train_data,id_map,ids,tokenizer,novel_index,batch_size=2)
    labels=[int(record["label"]>=0.5) for record in records]
    if sum(labels)!=len(episode["positive_ids"]) or len(labels)-sum(labels)!=len(episode["negative_ids"]):
        raise ValueError("Support episode is not exactly K positive + K negative")
    return records


def adapt_episode(base,train_data,id_map,episode,novel_label,novel_index,base_names,config,tokenizer):
    seed=int(episode["support_seed"]);records=encode_episode(base,train_data,id_map,episode,tokenizer,novel_index)
    labels=torch.tensor([r["label"] for r in records],dtype=torch.float32,device="cuda")
    h0=base.queries.detach().clone();initial_q,qstats=initialize_random_query(h0,42+novel_index*1000+seed+1)
    initial_w,initial_b=plain_scorer_and_bias(base,42+novel_index*1000+seed+2)
    w0,b0=parameter(initial_w),parameter(initial_b)
    q0=torch.nn.Parameter(initial_q.cuda())
    curves={};heads={}
    build0=lambda:(q0,w0,b0)
    curves["M0_random_query"]=fit_head(base,records,labels,build0,[q0,w0,b0],config,42+novel_index*1000+seed+3,"M0")
    heads["M0_random_query"]=to_cpu_head("M0_random_query",q0,w0,b0,{"random_query_stats":qstats})

    candidate_rows=[];candidate_heads=[]
    for j,name in enumerate(base_names):
        q=base.queries[j].detach().clone()
        w,b=parameter(initial_w),parameter(initial_b)
        build=lambda q=q,w=w,b=b:(q,w,b)
        curve=fit_head(base,records,labels,build,[w,b],config,42+novel_index*1000+seed+100+j,"M1_"+name)
        score=float(curve[-1]["support_loss"])
        candidate_rows.append({"source_label":name,"support_loss":score})
        candidate_heads.append(to_cpu_head("M1_best_old_query",q,w,b,{"source_label":name,"support_loss":score}))
    best_index=min(range(len(candidate_rows)),key=lambda j:candidate_rows[j]["support_loss"])
    heads["M1_best_old_query"]=candidate_heads[best_index]
    heads["M1_best_old_query"]["metadata"]["all_source_support_losses"]=candidate_rows
    curves["M1_best_old_query"]=[{"epoch":i+1,"support_loss":candidate_rows[best_index]["support_loss"]} for i in range(config["adapt_epochs"])]

    ap=torch.nn.Parameter(torch.zeros(len(base_names),device="cuda"));am=torch.nn.Parameter(torch.zeros(len(base_names),device="cuda"))
    w2,b2=parameter(initial_w),parameter(initial_b)
    build2=lambda:(query_mixture(base.queries,ap,am),w2,b2)
    curves["M2_old_query_mixture"]=fit_head(base,records,labels,build2,[ap,am,w2,b2],config,42+novel_index*1000+seed+4,"M2")
    qmix=query_mixture(base.queries,ap,am)
    alpha={"alpha_plus":torch.softmax(ap.detach(),0).cpu().tolist(),"alpha_minus":torch.softmax(am.detach(),0).cpu().tolist(),"old_label_order":base_names}
    heads["M2_old_query_mixture"]=to_cpu_head("M2_old_query_mixture",qmix,w2,b2,{"mixture":alpha,"source_support_losses":candidate_rows})

    basis3p,spanp=random_orthogonal_basis(base.queries.detach()[:,0,:],config["residual_rank"],42+novel_index*1000+seed+10)
    basis3m,spanm=random_orthogonal_basis(base.queries.detach()[:,1,:],config["residual_rank"],42+novel_index*1000+seed+11)
    basis3p=basis3p.cuda();basis3m=basis3m.cuda()
    cp=torch.nn.Parameter(torch.zeros(basis3p.shape[1],device="cuda"));cm=torch.nn.Parameter(torch.zeros(basis3m.shape[1],device="cuda"))
    qmix2=qmix.detach();w2f=w2.detach();b2f=b2.detach()
    build3=lambda:(torch.stack([qmix2[0]+basis3p@cp,qmix2[1]+basis3m@cm]),w2f,b2f)
    params3=[v for v in (cp,cm) if v.numel()]
    curves["M3_mixture_random_residual"]=fit_head(base,records,labels,build3,params3,config,42+novel_index*1000+seed+12,"M3") if params3 else []
    q3,_,_=build3()
    orth3p=float((spanp.T@basis3p.cpu()).norm()) if spanp.numel() and basis3p.numel() else 0.0
    orth3m=float((spanm.T@basis3m.cpu()).norm()) if spanm.numel() and basis3m.numel() else 0.0
    heads["M3_mixture_random_residual"]=to_cpu_head("M3_mixture_random_residual",q3,w2f,b2f,{"mixture":alpha,
        "residual_rank":max(basis3p.shape[1],basis3m.shape[1]),"rank_pos":basis3p.shape[1],"rank_neg":basis3m.shape[1],
        "R_pos":None,"R_neg":None,"C4_pos":None,"C4_neg":None,"orthogonality_pos":orth3p,"orthogonality_neg":orth3m})

    bgradp,bgradm,grad_diag=support_gradient_matrix(base,records,qmix2,w2f,b2f,float(config["auxiliary_weight"]))
    bgradp=bgradp.cuda();bgradm=bgradm.cuda()
    cgp=torch.nn.Parameter(torch.zeros(bgradp.shape[1],device="cuda"));cgm=torch.nn.Parameter(torch.zeros(bgradm.shape[1],device="cuda"))
    build4=lambda:(torch.stack([qmix2[0]+bgradp@cgp,qmix2[1]+bgradm@cgm]),w2f,b2f)
    params4=[v for v in (cgp,cgm) if v.numel()]
    curves["M4_mixture_gradient_residual"]=fit_head(base,records,labels,build4,params4,config,42+novel_index*1000+seed+13,"M4") if params4 else []
    q4,_,_=build4()
    diag4={"mixture":alpha,"residual_rank":max(bgradp.shape[1],bgradm.shape[1]),"rank_pos":bgradp.shape[1],"rank_neg":bgradm.shape[1],
           "R_pos":grad_diag["R_pos"],"R_neg":grad_diag["R_neg"],"C4_pos":grad_diag["C4_pos"],"C4_neg":grad_diag["C4_neg"],
           "orthogonality_pos":grad_diag["orthogonality_pos"],"orthogonality_neg":grad_diag["orthogonality_neg"]}
    heads["M4_mixture_gradient_residual"]=to_cpu_head("M4_mixture_gradient_residual",q4,w2f,b2f,diag4)

    m5,curve5=fit_m5(base,train_data,id_map,episode["positive_ids"]+episode["negative_ids"],tokenizer,novel_index,config,42+novel_index*1000+seed,initial_q,initial_w,initial_b)
    heads["M5_full_finetuning"]=m5
    curves["M5_full_finetuning"]=curve5
    checks={"alpha_plus_sum":float(sum(alpha["alpha_plus"])),"alpha_minus_sum":float(sum(alpha["alpha_minus"])),
        "M3_orthogonality_plus":orth3p,"M3_orthogonality_minus":orth3m,
        "M4_orthogonality_plus":grad_diag["orthogonality_pos"],"M4_orthogonality_minus":grad_diag["orthogonality_neg"],
        "M3_rank_max":int(max(basis3p.shape[1],basis3m.shape[1])),"M4_rank_max":int(max(bgradp.shape[1],bgradm.shape[1])),
        "base_model_parameter_hash_after_adaptation":model_hash(base)}
    return heads,curves,checks


def evaluate_frozen_heads(base,valid_data,heads,base_names,batch_size=4):
    device=torch.device("cuda");base.eval();order=sorted(range(len(valid_data)),key=valid_data.sequence_length)
    loader=DataLoader(torch.utils.data.Subset(valid_data,order),batch_size=batch_size,shuffle=False,num_workers=0,
        collate_fn=partial(collate_light_label_inference,pad_id=base.embedding.padding_idx),pin_memory=True)
    specs=[(label,seed,method,head) for label,seed,method,head in heads]
    all_queries=torch.cat([base.queries.detach().flatten(0,1).cpu()]+[item[3]["queries"].reshape(2,-1).cpu() for item in specs],0).to(device)
    all_scorers=torch.stack([item[3]["scorer"] for item in specs]).to(device)
    all_bias=torch.stack([item[3]["bias"] for item in specs]).to(device)
    predictions={i:[] for i in range(len(specs))}; old_logits=[];ids=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["input_ids"].to(device,non_blocking=True);mask=batch["mask"].to(device,non_blocking=True)
            q=all_queries.unsqueeze(0).expand(x.shape[0],-1,-1)
            with torch.autocast("cuda",dtype=torch.float16):
                hidden=base.encode_tokens(x,batch["lengths"],mask)
                evidence,_=base.cross_attention(q,hidden,hidden,key_padding_mask=~mask,need_weights=False)
                old_z=evidence[:,:base.num_labels*2].reshape(x.shape[0],base.num_labels,2,base.query_dim)
                old_score,_=base.score(old_z)
                new_z=evidence[:,base.num_labels*2:].reshape(x.shape[0],len(specs),2,base.query_dim)
                new_z=F.dropout(new_z,p=base.representation_dropout.p,training=False)
                energies=torch.einsum("bhpd,hd->bhp",new_z,all_scorers)+all_bias.unsqueeze(0)
                logits=energies[:,:,0]-energies[:,:,1]
            old_logits.append(old_score.float().cpu());ids.extend(batch["ids"])
            for j in range(len(specs)):predictions[j].append(logits[:,j].float().cpu())
    return {"ids":ids,"old_logits":torch.cat(old_logits).numpy(),
            "head_logits":{j:torch.cat(values).numpy() for j,values in predictions.items()},
            "head_index":{(label,seed,method):j for j,(label,seed,method,_) in enumerate(specs)}}


def evaluate_m5(base_state,head,valid_data,base_indices,base_names,config,batch_size=4):
    import copy
    device=torch.device("cuda")
    base=build_model("P11",dict(yaml.safe_load((ROOT/config["e3_base_config"]).read_text(encoding="utf-8")),
        label_names=base_names,num_labels=len(base_names),positive_auxiliary_label_multiplier=[1.0]*len(base_names),
        negative_auxiliary_label_multiplier=[1.0]*len(base_names)),
        len(EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])),
        EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"]).pad_token_id).to(device)
    base.load_state_dict(base_state,strict=True);base.eval()
    order=sorted(range(len(valid_data)),key=valid_data.sequence_length)
    loader=DataLoader(torch.utils.data.Subset(valid_data,order),batch_size=batch_size,shuffle=False,num_workers=0,
        collate_fn=partial(collate_light_label_inference,pad_id=base.embedding.padding_idx),pin_memory=True)
    q=torch.cat([base.queries.detach().flatten(0,1).cpu(),head["queries"].reshape(2,-1)],0).to(device)
    scorer=head["scorer"].to(device).unsqueeze(0);bias=head["bias"].to(device).unsqueeze(0)
    novel_logits=[];old_logits=[];ids=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["input_ids"].to(device,non_blocking=True);mask=batch["mask"].to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16):
                hidden=base.encode_tokens(x,batch["lengths"],mask)
                evidence,_=base.cross_attention(q.unsqueeze(0).expand(x.shape[0],-1,-1),hidden,hidden,key_padding_mask=~mask,need_weights=False)
                old_z=evidence[:,:base.num_labels*2].reshape(x.shape[0],base.num_labels,2,base.query_dim)
                old_score,_=base.score(old_z)
                z=evidence[:,base.num_labels*2:].reshape(x.shape[0],1,2,base.query_dim)
                z=F.dropout(z,p=base.representation_dropout.p,training=False)
                energies=torch.einsum("bhpd,hd->bhp",z,scorer)+bias.unsqueeze(0)
                logit=energies[:,:,0]-energies[:,:,1]
            ids.extend(batch["ids"]);old_logits.append(old_score.float().cpu());novel_logits.append(logit[:,0].float().cpu())
    del base
    return ids,torch.cat(old_logits).numpy(),torch.cat(novel_logits).numpy()


def metrics_for(labels,logits):
    return strict_metric(1/(1+np.exp(-np.asarray(logits))),labels)


def retention_metrics(labels,logits):
    y=np.asarray(labels,dtype=int);p=(1/(1+np.exp(-np.asarray(logits)))>=.5).astype(int)
    per=f1_score(y,p,average=None,zero_division=0)
    return {"macro_f1":float(np.mean(per)),"micro_f1":float(f1_score(y.reshape(-1),p.reshape(-1),zero_division=0)),
            "per_label_f1":[float(v) for v in per]}


def base_dev_macro_f1_from_audit(base_audit):
    return float(base_audit["base_dev_macro_f1_fixed05"])


def trainable_parameter_counts(method,head,base_config,base_model=None):
    d=int(base_config["query_dim"]);scorer=d+2
    if method=="M0_random_query":return {"query":2*d,"scorer":scorer,"encoder":0,"attention":0}
    if method=="M1_best_old_query":return {"query":0,"scorer":scorer,"encoder":0,"attention":0}
    if method=="M2_old_query_mixture":return {"query":2*len(base_config["label_names"]),"scorer":scorer,"encoder":0,"attention":0}
    if method in ("M3_mixture_random_residual","M4_mixture_gradient_residual"):
        return {"query":head["metadata"].get("rank_pos",head["metadata"].get("residual_rank",0))+head["metadata"].get("rank_neg",head["metadata"].get("residual_rank",0)),"scorer":0,"encoder":0,"attention":0}
    if method=="M5_full_finetuning":
        baseparams=base_model
        enc=sum(p.numel() for n,p in baseparams.named_parameters() if n.startswith(("embedding.","encoder.")))
        att=sum(p.numel() for n,p in baseparams.named_parameters() if n.startswith("cross_attention."))
        return {"query":2*d,"scorer":scorer,"encoder":enc,"attention":att}
    raise ValueError(method)


def assert_smoke(heads,checks,before_hash):
    if not np.isclose(checks["alpha_plus_sum"],1.0,atol=1e-6) or not np.isclose(checks["alpha_minus_sum"],1.0,atol=1e-6):
        raise ValueError("M2 alpha weights do not sum to one")
    for key,value in checks.items():
        if "orthogonality" in key and value>1e-4:raise ValueError(f"Residual basis is not orthogonal: {key}={value}")
        if "rank_max" in key and value>4:raise ValueError(f"Residual rank exceeded four: {key}={value}")
    if checks["base_model_parameter_hash_after_adaptation"]!=before_hash:
        raise ValueError("Frozen base model changed during query-only adaptation")
    if set(heads)!=set(METHODS):raise ValueError(f"Pilot methods missing: {set(METHODS)-set(heads)}")
    for name,head in heads.items():
        if not torch.isfinite(head["queries"]).all() or not torch.isfinite(head["scorer"]).all():
            raise ValueError(f"Nonfinite head in {name}")


def load_valid_labels(config,valid_dataset):
    rows=read_valid_labels(ROOT/config["dataset"]/"valid.jsonl")
    if [x[0] for x in rows]!=valid_dataset.ids:
        raise ValueError("Final-evaluation valid labels do not align with the ID-only opcode cache")
    return {cid:label for cid,label in rows}


def write_episode_row(path,row,fieldnames):
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
        if not exists:writer.writeheader()
        writer.writerow(row)


EPISODE_FIELDS=["novel_label","K","support_seed","method","support_positive_ids","support_negative_ids",
    "novel_valid_positive_count","novel_valid_negative_count","pr_auc","roc_auc","f1_05","precision_05","recall_05",
    "old_macro_f1_before","old_macro_f1_after","old_micro_f1_before","old_micro_f1_after",
    "trainable_query_params","trainable_scorer_params","trainable_encoder_params","total_trainable_params",
    "best_source_label","residual_rank","R_pos","R_neg","C4_pos","C4_neg"]


def base_old_metrics(labels,logits):
    return retention_metrics(labels,logits)


def write_method_outputs(config,episode_rows,alpha_rows,residual_rows,param_rows,retention_rows,curves_rows):
    root=ROOT/config["result_root"];root.mkdir(parents=True,exist_ok=True)
    products=(("episode_results.csv",episode_rows),("alpha_weights.csv",alpha_rows),
        ("residual_diagnostics.csv",residual_rows),("parameter_counts.csv",param_rows),
        ("old_label_retention.csv",retention_rows),("support_loss_curves.csv",curves_rows))
    for filename,rows in products:
        path=root/filename
        if not rows:continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def final_evaluate_label(novel_label,split,base,base_config,base_indices,base_audit,episode_states,train_full,valid_dataset,config,tokenizer,valid_labels):
    novel_index=split["novel_label_index"];base_names=split["base_labels"]
    # All frozen-encoder variants and the base reference share one encoder pass per valid batch.
    frozen_specs=[]
    for seed,states in episode_states.items():
        for method in METHODS[:5]:
            frozen_specs.append((novel_label,int(seed),method,states["heads"][method]))
    frozen_eval=evaluate_frozen_heads(base,valid_dataset,frozen_specs,base_names,batch_size=4)
    valid_ids=frozen_eval["ids"]
    y_all=np.asarray([valid_labels[cid] for cid in valid_ids],dtype=int)
    y_novel=y_all[:,novel_index]
    y_old=y_all[:,base_indices]
    old_before=base_old_metrics(y_old,frozen_eval["old_logits"])
    valid_pos=int(y_novel.sum());valid_neg=int(len(y_novel)-valid_pos)
    rows=[];alpha_rows=[];residual_rows=[];param_rows=[];retention_rows=[]
    head_idx=frozen_eval["head_index"]
    for seed,states in episode_states.items():
        episode=states["episode"]
        for method in METHODS[:5]:
            head=states["heads"][method]
            index=head_idx[(novel_label,int(seed),method)]
            metric=metrics_for(y_novel,frozen_eval["head_logits"][index])
            meta=head.get("metadata",{})
            params=trainable_parameter_counts(method,head,base_config)
            total=sum(params.values())
            row={"novel_label":novel_label,"K":config["support_k"],"support_seed":int(seed),"method":method,
                "support_positive_ids":";".join(episode["positive_ids"]),"support_negative_ids":";".join(episode["negative_ids"]),
                "novel_valid_positive_count":valid_pos,"novel_valid_negative_count":valid_neg,**metric,
                "old_macro_f1_before":old_before["macro_f1"],"old_macro_f1_after":old_before["macro_f1"],
                "old_micro_f1_before":old_before["micro_f1"],"old_micro_f1_after":old_before["micro_f1"],
                "trainable_query_params":params["query"],"trainable_scorer_params":params["scorer"],
                "trainable_encoder_params":params["encoder"],"total_trainable_params":total,
                "best_source_label":meta.get("source_label"),"residual_rank":meta.get("residual_rank"),
                "R_pos":meta.get("R_pos"),"R_neg":meta.get("R_neg"),"C4_pos":meta.get("C4_pos"),"C4_neg":meta.get("C4_neg")}
            rows.append(row);param_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":method,**params,"total":total})
            retention_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":method,
                "old_macro_f1_before":old_before["macro_f1"],"old_macro_f1_after":old_before["macro_f1"],
                "old_micro_f1_before":old_before["micro_f1"],"old_micro_f1_after":old_before["micro_f1"],"delta_macro":0.0,"delta_micro":0.0})
            if "mixture" in meta:
                weights=meta["mixture"]
                alpha_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":method,
                    **{f"alpha_plus_{name}":value for name,value in zip(weights["old_label_order"],weights["alpha_plus"])},
                    **{f"alpha_minus_{name}":value for name,value in zip(weights["old_label_order"],weights["alpha_minus"])}})
            if method in ("M3_mixture_random_residual","M4_mixture_gradient_residual"):
                residual_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":method,"pr_auc":metric["pr_auc"],
                    **{k:meta.get(k) for k in ("R_pos","R_neg","C4_pos","C4_neg","residual_rank","rank_pos","rank_neg","orthogonality_pos","orthogonality_neg")}})
        # M5 uses its own encoder and is evaluated in a separate pass.
        m5=states["heads"]["M5_full_finetuning"]
        state_path=ROOT/config["checkpoint_root"]/novel_label.replace(" ","_")/f"seed_{seed}"/"M5_shared_state.pt"
        m5_state=torch.load(state_path,map_location="cpu",weights_only=False)
        ids5,old_logits5,novel_logits5=evaluate_m5(m5_state,m5,valid_dataset,base_indices,base_names,config,batch_size=1)
        y5=np.asarray([valid_labels[cid] for cid in ids5],dtype=int)
        if ids5!=valid_ids:
            raise ValueError("M5 valid ID order differs from shared validation evaluation")
        old_after=base_old_metrics(y5[:,base_indices],old_logits5)
        metric5=metrics_for(y5[:,novel_index],novel_logits5)
        params=m5.get("parameter_counts",{})
        row={"novel_label":novel_label,"K":config["support_k"],"support_seed":int(seed),"method":"M5_full_finetuning",
            "support_positive_ids":";".join(states["episode"]["positive_ids"]),"support_negative_ids":";".join(states["episode"]["negative_ids"]),
            "novel_valid_positive_count":valid_pos,"novel_valid_negative_count":valid_neg,**metric5,
            "old_macro_f1_before":old_before["macro_f1"],"old_macro_f1_after":old_after["macro_f1"],
            "old_micro_f1_before":old_before["micro_f1"],"old_micro_f1_after":old_after["micro_f1"],
            "trainable_query_params":params.get("query",0),"trainable_scorer_params":params.get("scorer",0),
            "trainable_encoder_params":params.get("encoder",0),"total_trainable_params":sum(params.values()),
            "best_source_label":None,"residual_rank":None,"R_pos":None,"R_neg":None,"C4_pos":None,"C4_neg":None}
        rows.append(row);param_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":"M5_full_finetuning",**params,"total":sum(params.values())})
        retention_rows.append({"novel_label":novel_label,"support_seed":int(seed),"method":"M5_full_finetuning",
            "old_macro_f1_before":old_before["macro_f1"],"old_macro_f1_after":old_after["macro_f1"],
            "old_micro_f1_before":old_before["micro_f1"],"old_micro_f1_after":old_after["micro_f1"],
            "delta_macro":old_after["macro_f1"]-old_before["macro_f1"],"delta_micro":old_after["micro_f1"]-old_before["micro_f1"]})
    base_report={"novel_label":novel_label,"best_epoch":base_audit["best_epoch"],"base_dev_old_macro_f1_fixed05":base_dev_macro_f1_from_audit(base_audit),
        "official_valid_old_macro_f1_before":old_before["macro_f1"],"official_valid_old_micro_f1_before":old_before["micro_f1"],
        "valid_positive_count":valid_pos,"valid_negative_count":valid_neg,"test_checked":False}
    (ROOT/config["result_root"]/novel_label.replace(" ","_")/"final_eval.json").write_text(json.dumps(base_report,indent=2),encoding="utf-8")
    return rows,alpha_rows,residual_rows,param_rows,retention_rows,old_before


def load_base_checkpoint(label,config,tokenizer):
    base_config=yaml.safe_load((ROOT/config["e3_base_config"]).read_text(encoding="utf-8"))
    novel_index=base_config["label_names"].index(label);indices=[i for i in range(8) if i!=novel_index]
    base_config["label_names"]=[base_config["label_names"][i] for i in indices];base_config["num_labels"]=7
    base_config["positive_auxiliary_label_multiplier"]=[1.0]*7;base_config["negative_auxiliary_label_multiplier"]=[1.0]*7
    path=ROOT/config["checkpoint_root"]/label.replace(" ","_")/"base_seed42"/"best.pt"
    saved=torch.load(path,map_location="cpu",weights_only=False)
    model=build_model("P11",base_config,len(tokenizer),tokenizer.pad_token_id).cuda()
    model.load_state_dict(saved["model_state_dict"],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    audit=json.loads((ROOT/config["result_root"]/label.replace(" ","_")/"base_seed42"/"base_audit.json").read_text(encoding="utf-8"))
    return model,base_config,indices,audit


def save_episode(novel_label,episode,heads,curves,checks,config):
    root=ROOT/config["result_root"]/novel_label.replace(" ","_")/f"seed_{episode['support_seed']}"
    ckpt=ROOT/config["checkpoint_root"]/novel_label.replace(" ","_")/f"seed_{episode['support_seed']}"
    root.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    m5=heads["M5_full_finetuning"]
    m5_state=m5["metadata"].pop("fine_tuned_state")
    base_cfg=yaml.safe_load((ROOT/config["e3_base_config"]).read_text(encoding="utf-8"))
    base_cfg["label_names"]=[name for name in base_cfg["label_names"] if name!=novel_label];base_cfg["num_labels"]=7
    base_cfg["positive_auxiliary_label_multiplier"]=[1.0]*7;base_cfg["negative_auxiliary_label_multiplier"]=[1.0]*7
    tok=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"]);base=build_model("P11",base_cfg,len(tok),tok.pad_token_id)
    all_shared=sum(p.numel() for n,p in base.named_parameters() if n.startswith(("embedding.","encoder.","cross_attention.")))
    encoder=sum(p.numel() for n,p in base.named_parameters() if n.startswith(("embedding.","encoder.")))
    attention=all_shared-encoder
    m5["metadata"]["parameter_counts"]={"query":m5["metadata"]["trainable_query_params"],
        "scorer":m5["metadata"]["trainable_scorer_params"],"encoder":encoder,"attention":attention}
    torch.save(m5_state,ckpt/"M5_shared_state.pt")
    torch.save({"signature":digest({"label":novel_label,"seed":episode["support_seed"],"episode":episode,"config":config}),
        "episode":episode,"heads":heads,"curves":curves,"checks":checks,"test_checked":False},root/"adaptation_state.pt")
    curve_rows=[]
    for method,curve in curves.items():
        for row in curve:curve_rows.append({"novel_label":novel_label,"support_seed":episode["support_seed"],"method":method,**row})
    (root/"support_loss_curves.json").write_text(json.dumps(curve_rows,indent=2),encoding="utf-8")
    (root/"checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    return {"signature":digest({"label":novel_label,"seed":episode["support_seed"],"episode":episode,"config":config}),
        "episode":episode,"heads":heads,"curves":curves,"checks":checks,"test_checked":False}


def load_or_train_episode(base,label,split,episode,train_data,id_map,config,tokenizer):
    path=ROOT/config["result_root"]/label.replace(" ","_")/f"seed_{episode['support_seed']}"/"adaptation_state.pt"
    signature=digest({"label":label,"seed":episode["support_seed"],"episode":episode,"config":config})
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=False)
        if state.get("signature")==signature:
            return state
    label_names=yaml.safe_load((ROOT/config["e3_base_config"]).read_text(encoding="utf-8"))["label_names"]
    novel_index=label_names.index(label);base_names=[name for name in label_names if name!=label]
    heads,curves,checks=adapt_episode(base,train_data,id_map,episode,label,novel_index,base_names,config,tokenizer)
    state=save_episode(label,episode,heads,curves,checks,config)
    return state


def export_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def summarize_and_plot(config,episode_rows,alpha_rows,residual_rows,param_rows,retention_rows,curve_rows):
    import matplotlib.pyplot as plt
    root=ROOT/config["result_root"];reports=ROOT/config["report_root"];plots=reports/"plots"
    root.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    export_csv(root/"episode_results.csv",episode_rows);export_csv(root/"alpha_weights.csv",alpha_rows)
    export_csv(root/"residual_diagnostics.csv",residual_rows);export_csv(root/"parameter_counts.csv",param_rows)
    export_csv(root/"old_label_retention.csv",retention_rows);export_csv(root/"support_loss_curves.csv",curve_rows)
    means={}
    for method in METHODS:
        vals=[row["pr_auc"] for row in episode_rows if row["method"]==method and row.get("pr_auc") is not None]
        means[method]={"mean_pr_auc":float(np.mean(vals)) if vals else None,"std_pr_auc":float(np.std(vals)) if vals else None,"episodes":len(vals)}
    per_label={}
    for label in config["heldout_labels"]:
        row={"M0":[],"M2":[],"M3":[],"M4":[]}
        for episode in episode_rows:
            if episode["novel_label"]!=label:continue
            key={"M0_random_query":"M0","M2_old_query_mixture":"M2","M3_mixture_random_residual":"M3","M4_mixture_gradient_residual":"M4"}.get(episode["method"])
            if key:row[key].append(episode["pr_auc"])
        per_label[label]={key:(float(np.mean(vals)) if vals else None) for key,vals in row.items()}
    h1={label:per_label[label]["M2"]-per_label[label]["M0"] for label in config["heldout_labels"] if per_label[label]["M2"] is not None and per_label[label]["M0"] is not None}
    h2_m2={label:per_label[label]["M4"]-per_label[label]["M2"] for label in config["heldout_labels"] if per_label[label]["M4"] is not None and per_label[label]["M2"] is not None}
    h2_m3={label:per_label[label]["M4"]-per_label[label]["M3"] for label in config["heldout_labels"] if per_label[label]["M4"] is not None and per_label[label]["M3"] is not None}
    h1_ok=sum(v>0 for v in h1.values());h2_ok=sum((h2_m2.get(label,-1)>0 and h2_m3.get(label,-1)>0) for label in config["heldout_labels"])
    decision="GO" if h1_ok>=3 and h2_ok>=3 else "CONDITIONAL" if h1_ok>=3 and h2_ok>=2 else "NO-GO"
    fig,ax=plt.subplots(figsize=(11,5));labels=list(METHODS);values=[means[m]["mean_pr_auc"] for m in labels]
    errors=[means[m]["std_pr_auc"] for m in labels];ax.bar(range(len(labels)),values,yerr=errors,capsize=3,color="#457B9D")
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha="right");ax.set_ylabel("Average Precision (valid)");ax.set_title("Few-shot novel-label methods (K=10)");fig.tight_layout();fig.savefig(plots/"method_pr_auc.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ax,label in zip(axes.flat,config["heldout_labels"]):
        for method,color in (("M2_old_query_mixture","#457B9D"),("M3_mixture_random_residual","#E9C46A"),("M4_mixture_gradient_residual","#2A9D8F")):
            rows=sorted([r for r in episode_rows if r["novel_label"]==label and r["method"]==method],key=lambda x:x["support_seed"])
            if rows:ax.plot([r["support_seed"] for r in rows],[r["pr_auc"] for r in rows],marker="o",label=method,color=color)
        ax.set_title(label);ax.set_xlabel("Support seed");ax.set_ylabel("Average Precision")
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(plots/"seedwise_residual_gain.png",dpi=180);plt.close(fig)
    for key,title,outname in (("alpha_plus","Positive query mixture","alpha_heatmap_positive.png"),("alpha_minus","Negative query mixture","alpha_heatmap_negative.png")):
        rows=[r for r in alpha_rows if r["method"]=="M2_old_query_mixture"]
        old_names=[name for name in config["label_names"] if any(f"{key}_{name}" in row for row in rows)]
        matrix=np.asarray([[row.get(f"{key}_{name}",0.0) for name in old_names] for row in rows])
        fig,ax=plt.subplots(figsize=(10,max(4,len(rows)*.23)));im=ax.imshow(matrix,aspect="auto",cmap="viridis",vmin=0)
        ax.set_xticks(range(len(old_names)),old_names,rotation=35,ha="right");ax.set_yticks(range(len(rows)),[f"{r['novel_label']} s{r['support_seed']}" for r in rows],fontsize=7)
        ax.set_title(title);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(plots/outname,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for label in config["heldout_labels"]:
        group=[r for r in residual_rows if r["novel_label"]==label and r["method"]=="M4_mixture_gradient_residual" and r.get("R_pos") is not None]
        if group:
            gains=[]
            for r in group:
                m2=next((x["pr_auc"] for x in episode_rows if x["novel_label"]==label and x["support_seed"]==r["support_seed"] and x["method"]=="M2_old_query_mixture"),None)
                m4=next((x["pr_auc"] for x in episode_rows if x["novel_label"]==label and x["support_seed"]==r["support_seed"] and x["method"]=="M4_mixture_gradient_residual"),None)
                if m2 is not None and m4 is not None:gains.append((.5*(r["R_pos"]+r["R_neg"]),m4-m2))
            if gains:ax.scatter([x for x,_ in gains],[y for _,y in gains],label=label)
    ax.axhline(0,color="black",linewidth=.8);ax.set_xlabel("Mean residual gradient ratio");ax.set_ylabel("M4 PR-AUC − M2 PR-AUC");ax.legend();fig.tight_layout();fig.savefig(plots/"residual_ratio_vs_gain.png",dpi=180);plt.close(fig)
    summary={"route":"Few-shot Novel Vulnerability Query Expansion Pilot-0","dataset":config["dataset"],"official_test_accessed":False,
        "base_training_seed":config["base_seed"],"support_k":config["support_k"],"support_seeds":config["support_seeds"],
        "methods":means,"per_label_mean_pr_auc":per_label,"H1_mixture_minus_random_by_label":h1,
        "H2_gradient_residual_minus_mixture_by_label":h2_m2,"H2_gradient_residual_minus_random_residual_by_label":h2_m3,
        "H1_positive_labels":h1_ok,"H2_positive_labels":h2_ok,"decision":decision}
    (root/"pilot_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Few-shot Novel Vulnerability Query Expansion Pilot-0","",f"Dataset: `{config['dataset']}`. Official test content/predictions were not accessed. Base seed={config['base_seed']}; K={config['support_k']}; support seeds={config['support_seeds']}.","",
        "## Results","","| Method | Mean PR-AUC | SD | Episodes |","|---|---:|---:|---:|"]
    for method,v in means.items():lines.append(f"| {method} | {v['mean_pr_auc'] if v['mean_pr_auc'] is not None else float('nan'):.4f} | {v['std_pr_auc'] if v['std_pr_auc'] is not None else float('nan'):.4f} | {v['episodes']} |")
    lines += ["","| Novel label | M0 random | M2 mixture | M3 random residual | M4 gradient residual | M2−M0 | M4−M2 | M4−M3 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in config["heldout_labels"]:
        v=per_label[label];lines.append(f"| {label} | {v['M0'] or float('nan'):.4f} | {v['M2'] or float('nan'):.4f} | {v['M3'] or float('nan'):.4f} | {v['M4'] or float('nan'):.4f} | {h1.get(label,float('nan')):.4f} | {h2_m2.get(label,float('nan')):.4f} | {h2_m3.get(label,float('nan')):.4f} |")
    lines += ["","## Decision","",f"**{decision}**. H1 positive label count={h1_ok}/4; H2 positive label count={h2_ok}/4. Decision uses only the predeclared across-label direction criteria. No threshold was tuned on official valid.",""]
    (ROOT/config["report_root"]/"pilot_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true",help="Run Reentrancy seed 0 M0-M5 support-only smoke; does not evaluate official valid")
    parser.add_argument("--full",action="store_true",help="Run all four held-out labels x five support seeds, then one final valid evaluation")
    parser.add_argument("--allow-test",action="store_true",help="Test access is prohibited by this pilot")
    args=parser.parse_args()
    if args.allow_test:raise ValueError("Official test is locked; --allow-test is prohibited")
    if args.smoke==args.full:raise ValueError("Select exactly one of --smoke or --full")
    config=yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["allow_test"] is not False:raise ValueError("Config must keep allow_test=false")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":raise RuntimeError("learnDL310 CUDA is required")
    tok=EVMOpcodeTokenizer.from_vocab_file(ROOT/config["vocab_path"])
    train_full=LightLabelDataset(ROOT/config["cache_dir"]/f"train_max{config['max_len']}.pt",runtime_max_len=config["max_len"])
    idmap={str(cid):i for i,cid in enumerate(train_full.ids)}
    reports=ROOT/config["report_root"];reports.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=ROOT/config["checkpoint_root"]
    pending_labels=["Reentrancy"] if args.smoke else config["heldout_labels"]
    pending_seeds=[0] if args.smoke else config["support_seeds"]
    completed={}
    for label in pending_labels:
        split=read_split_manifest(label)
        base,base_config,base_indices,base_audit=train_base(label,config,split,train_full,tok)
        base_before=model_hash(base)
        label_states={}
        for episode in split["episodes"]:
            if int(episode["support_seed"]) not in pending_seeds:continue
            state=load_or_train_episode(base,label,split,episode,train_full,idmap,config,tok)
            checks=state["checks"];assert_smoke(state["heads"],checks,base_before)
            label_states[int(episode["support_seed"])]=state
        completed[label]={"split":split,"base_audit":base_audit,"states":label_states}
        print(f"[pilot] {label} base_epoch={base_audit['best_epoch']} episodes={len(label_states)} support_fit_complete",flush=True)
    if args.smoke:
        state=completed["Reentrancy"]["states"][0]
        smoke={"label":"Reentrancy","support_seed":0,"methods":{m:{"support_last_loss":state["curves"][m][-1]["support_loss"] if state["curves"][m] else None,
            "query_shape":list(state["heads"][m]["queries"].shape)} for m in METHODS},"checks":state["checks"],
            "official_valid_used":False,"official_test_accessed":False}
        (reports/"reentrancy_smoke.json").write_text(json.dumps(smoke,indent=2),encoding="utf-8")
        print(json.dumps(smoke,indent=2),flush=True);return
    # Official valid labels are first read here, after every base checkpoint and support adaptation is fixed.
    valid_cache=OpcodeOnlyDataset(ROOT/config["cache_dir"]/f"valid_max{config['max_len']}.pt",config["max_len"])
    valid_rows=read_valid_labels(ROOT/config["dataset"]/"valid.jsonl")
    if [cid for cid,_ in valid_rows]!=valid_cache.ids:raise ValueError("Official valid ID order mismatches token cache")
    valid_label_map=dict(valid_rows)
    all_episode_rows=[];all_alpha=[];all_residual=[];all_params=[];all_retention=[];all_curves=[]
    for label in config["heldout_labels"]:
        split=read_split_manifest(label)
        base,base_config,base_indices,base_audit=load_base_checkpoint(label,config,tok)
        states={}
        for episode in split["episodes"]:
            ep_dir=ROOT/config["result_root"]/label.replace(" ","_")/f"seed_{episode['support_seed']}"
            payload=torch.load(ep_dir/"adaptation_state.pt",map_location="cpu",weights_only=False)
            states[int(episode["support_seed"])]=payload
            state_map=payload["heads"]
            for method in METHODS[:5]:
                head=state_map[method];meta=head.get("metadata",{})
                all_curves.extend({"novel_label":label,"support_seed":episode["support_seed"],"method":method,**r} for r in payload["curves"][method])
                if method=="M2_old_query_mixture":
                    alpha=meta["mixture"]
                    all_alpha.append({"novel_label":label,"support_seed":episode["support_seed"],"method":method,
                        **{f"alpha_plus_{n}":v for n,v in zip(alpha["old_label_order"],alpha["alpha_plus"])},
                        **{f"alpha_minus_{n}":v for n,v in zip(alpha["old_label_order"],alpha["alpha_minus"])}})
                if method in ("M3_mixture_random_residual","M4_mixture_gradient_residual"):
                    all_residual.append({"novel_label":label,"support_seed":episode["support_seed"],"method":method,**{k:meta.get(k) for k in (
                        "R_pos","R_neg","C4_pos","C4_neg","residual_rank","rank_pos","rank_neg","orthogonality_pos","orthogonality_neg")}})
        erows,arows,rrows,prows,retrows,_=final_evaluate_label(label,split,base,base_config,base_indices,base_audit,states,train_full,valid_cache,config,tok,valid_label_map)
        all_episode_rows.extend(erows);all_params.extend(prows);all_retention.extend(retrows)
        print(f"[pilot:valid] {label} episodes={len(states)} done; no threshold tuning",flush=True)
    write_method_outputs(config,all_episode_rows,all_alpha,all_residual,all_params,all_retention,all_curves)
    summary=summarize_and_plot(config,all_episode_rows,all_alpha,all_residual,all_params,all_retention,all_curves)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
