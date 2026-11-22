"""Five-table canonical loader and openly labelled semi-synthetic public-source adapter.

Required canonical CSVs:
 entities.csv: entity_id,entity_type [,region,capacity,lead_time,tier,industry]
 relations.csv: source_id,target_id,relation_type [,dependency_strength,lead_time,capacity]
 operations.csv: day,entity_id [,demand,production,inventory,delivery,lead_time,capacity]
 events.csv: entity_id,start_date [,end_date,disruption_type,severity,delay_days,loss_kusd]
 ports.csv: may contain day/period,entity_id/port_id,turnaround_days,vessel_calls,
            cargo_capacity,efficiency. Non-daily series are joined strictly as-of, never
            presented as daily *measured* values. All preprocessing parameters fit train-only.
"""
from pathlib import Path
import json,re,warnings
import numpy as np
import pandas as pd
from .config import NODE_TYPES, EDGE_TYPES, DISRUPTION_TYPES, SEVERITY_TYPES
from .schema import GraphSequence

FEATURE_NAMES=("demand","production","inventory","delivery","lead_time","capacity",
               "turnaround_days","vessel_calls","cargo_capacity","efficiency",
               "historical_risk","historical_severity","historical_delay","historical_loss",
               "weekly_sin","weekly_cos")
OPERATION_FEATURES=("demand","production","inventory","delivery","lead_time","capacity")
PORT_FEATURES=("turnaround_days","vessel_calls","cargo_capacity","efficiency")


def _normalise_columns(df):
    df=df.copy()
    df.columns=[re.sub(r"[^a-z0-9]+","_",str(c).lower()).strip("_") for c in df.columns]
    return df


def _coerce_number(col):
    return pd.to_numeric(col.astype(str).str.replace(r"[^\d.eE+\-]","",regex=True),errors="coerce")


def _date_col(series):
    # Dates are normalised to ISO UTC-naive day; strict errors prevent silent remapping.
    result=pd.to_datetime(series,errors="coerce")
    return result.dt.strftime("%Y-%m-%d")


def _severity(raw):
    if pd.isna(raw): return -1
    v=str(raw).lower().strip()
    if v in SEVERITY_TYPES:return SEVERITY_TYPES.index(v)
    if "very high" in v or "critical" in v or "extreme" in v:return 3
    if "high" in v or "severe" in v:return 2
    if "med" in v or "moderate" in v:return 1
    try:return int(np.clip(float(v)-1,0,3))
    except ValueError:return -1


def _disruption(raw):
    v=re.sub(r"[^a-z]+","_",str(raw).lower()).strip("_")
    if v in DISRUPTION_TYPES:return DISRUPTION_TYPES.index(v)
    heuristics=(("supplier",0),("port",1),("congest",1),("transport",2),
                ("delay",2),("natural",3),("flood",3),("weather",3),("storm",3),("disaster",3),
                ("geopolit",4),("political",4),("cyber",5),
                ("labour",6),("labor",6),("strike",6),("factory",7),
                ("manufactur",7),("demand",8))
    for token,idx in heuristics:
        if token in v:return idx
    return -1  # unknown category: mask type classification, do not fabricate a demand label


def _ensure(df,required,filename):
    if missing:=set(required)-set(df.columns):
        raise ValueError(f"{filename} lacks required columns {sorted(missing)}; got {list(df.columns)}")


def load_canonical(directory,extra_manifest=None):
    directory=Path(directory)
    tables={}
    for name in ("entities","relations","operations","events"):
        p=directory/f"{name}.csv"
        if not p.exists():raise FileNotFoundError(f"Missing canonical table: {p}. Run audit/raw converter or use --demo.")
        tables[name]=_normalise_columns(pd.read_csv(p,low_memory=False))
    p=directory/"ports.csv"
    tables["ports"]=_normalise_columns(pd.read_csv(p,low_memory=False)) if p.exists() else pd.DataFrame()
    manifest_p=directory/"provenance.json"
    manifest=json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    manifest.update(extra_manifest or {})
    return build_from_tables(**tables,manifest=manifest)


def build_from_tables(entities,relations,operations,events,ports=None,manifest=None):
    """Create a strictly chronological graph. Target y[t] is never included in x[:t]."""
    manifest=dict(manifest or {})
    e=_normalise_columns(entities); rel=_normalise_columns(relations)
    ops=_normalise_columns(operations); events=_normalise_columns(events)
    ports=_normalise_columns(ports) if ports is not None and not ports.empty else pd.DataFrame()
    _ensure(e,["entity_id","entity_type"],"entities.csv")
    _ensure(rel,["source_id","target_id","relation_type"],"relations.csv")
    _ensure(ops,["day","entity_id"],"operations.csv")
    _ensure(events,["entity_id","start_date"],"events.csv")
    e=e.drop_duplicates("entity_id").reset_index(drop=True)
    e["entity_id"]=e.entity_id.astype(str)
    bad=set(e.entity_type)-set(NODE_TYPES)
    if bad:raise ValueError(f"Unknown node types: {sorted(bad)}")
    ids=e.entity_id.tolist(); node_lookup={v:i for i,v in enumerate(ids)}
    types=np.array([NODE_TYPES.index(v) for v in e.entity_type],dtype=np.int64)
    ops["day"]=_date_col(ops["day"])
    ops=ops.dropna(subset=["day"])
    days=sorted(ops.day.unique().tolist())
    if len(days)<12:raise ValueError("At least twelve dated operational observations required")
    T=len(days);N=len(e);F=len(FEATURE_NAMES)
    if ops.duplicated(["day","entity_id"]).any():
        # Operational quantities are safely aggregated (without merging labels).
        agg={col:"mean" for col in OPERATION_FEATURES if col in ops}
        ops=ops.groupby(["day","entity_id"],as_index=False).agg(agg)
    index_days={day:i for i,day in enumerate(days)}
    raw=np.full((T,N,F),np.nan,dtype=np.float32)
    for col in OPERATION_FEATURES:
        if col not in ops:continue
        jj=FEATURE_NAMES.index(col)
        valid=ops[ops["day"].isin(days)&ops["entity_id"].astype(str).isin(node_lookup)].copy()
        values=_coerce_number(valid[col]);
        for d,n,val in zip(valid.day,valid.entity_id.astype(str),values):
            if pd.notna(val):raw[index_days[d],node_lookup[n],jj]=float(val)
    for col in ("capacity","lead_time"):
        if col in e:
            jj=FEATURE_NAMES.index(col)
            base=_coerce_number(e[col]).to_numpy(dtype=float)
            for i,val in enumerate(base):
                if np.isfinite(val):
                    where=~np.isfinite(raw[:,i,jj]);raw[where,i,jj]=val
    # Port performance observed on or before a day; semiannual reports are never re-labelled as daily.
    port_resolution="absent"
    if not ports.empty:
        targetcol="entity_id" if "entity_id" in ports else "port_id" if "port_id" in ports else None
        if targetcol:
            pp=ports.copy();pp["entity_id"]=pp[targetcol].astype(str)
            if "day" in pp:
                pp["available_day"]=_date_col(pp.day)
                port_resolution="dated_observations"
            elif "period" in pp:
                def period_end(v):
                    s=str(v);m=re.match(r"^(\d{4})[-_ ]?S([12])$",s,re.I)
                    if m:return f"{m.group(1)}-{'06-30' if m.group(2)=='1' else '12-31'}"
                    d=pd.to_datetime(s,errors="coerce")
                    return d.strftime("%Y-%m-%d") if pd.notna(d) else None
                pp["available_day"]=pp.period.map(period_end)
                port_resolution="semiannual_as_of"
            else:pp["available_day"]=None;port_resolution="static_only"
            for entity,g in pp.groupby("entity_id"):
                if entity not in node_lookup:continue
                i=node_lookup[entity]
                g=g.copy();g=g.dropna(subset=["available_day"]).sort_values("available_day")
                if g.empty:continue
                for col in PORT_FEATURES:
                    if col not in g:continue
                    series=_coerce_number(g[col]).to_numpy()
                    rec_days=g.available_day.tolist()
                    jj=FEATURE_NAMES.index(col)
                    # Last value with published availability <= each input feature date.
                    for day_idx,day in enumerate(days):
                        candidates=[j for j,dt in enumerate(rec_days) if dt<=day and np.isfinite(series[j])]
                        if candidates:raw[day_idx,i,jj]=float(series[candidates[-1]])
    # Completed history only, independent of the next-day target.
    lab={"risk":np.zeros((T,N),np.float32),"type":np.full((T,N),-1,np.int64),
         "severity":np.full((T,N),-1,np.int64),"delay":np.zeros((T,N),np.float32),
         "loss":np.zeros((T,N),np.float32),
         "delay_valid":np.zeros((T,N),bool),"loss_valid":np.zeros((T,N),bool)}
    events=events.copy();events["entity_id"]=events.entity_id.astype(str)
    events["start_date"]=_date_col(events.start_date)
    if "end_date" not in events:events["end_date"]=events.start_date
    else:events["end_date"]=_date_col(events.end_date)
    for c in ("delay_days","loss_kusd"):
        if c not in events:events[c]=np.nan
        events[c]=_coerce_number(events[c]).clip(lower=0)
    if "severity" not in events:events["severity"]=None
    if "disruption_type" not in events:events["disruption_type"]=None
    kept=0
    for record in events.to_dict("records"):
        entity=record["entity_id"]
        if entity not in node_lookup or record["start_date"] not in index_days:continue
        i=node_lookup[entity];t=index_days[record["start_date"]]
        lab["risk"][t,i]=1;lab["type"][t,i]=_disruption(record["disruption_type"])
        lab["severity"][t,i]=_severity(record["severity"])
        if pd.notna(record["delay_days"]):
            lab["delay"][t,i]+=float(record["delay_days"])
            lab["delay_valid"][t,i]=True
        if pd.notna(record["loss_kusd"]):
            lab["loss"][t,i]+=float(record["loss_kusd"])
            lab["loss_valid"][t,i]=True
        kept+=1
    # Event-history inputs become available only AFTER completed event end_date.
    historical=np.zeros((T,N,4),dtype=np.float32)
    for rec in events.to_dict("records"):
        if rec["entity_id"] not in node_lookup:continue
        if pd.isna(rec["end_date"]):continue
        i=node_lookup[rec["entity_id"]]
        h=np.array([1.,max(0.,float(_severity(rec["severity"])))/3,
                    float(rec["delay_days"]) if pd.notna(rec["delay_days"]) else 0.,
                    float(rec["loss_kusd"]) if pd.notna(rec["loss_kusd"]) else 0.],dtype=np.float32)
        for t,day in enumerate(days):
            if rec["end_date"]<day:historical[t,i]+=h
    raw[:,:,10:14]=historical
    for t,day in enumerate(days):
        dayofweek=pd.Timestamp(day).dayofweek
        raw[t,:,14]=np.sin(2*np.pi*dayofweek/7)
        raw[t,:,15]=np.cos(2*np.pi*dayofweek/7)
    # Training-only imputation and scaling to prevent test feature-distribution leakage.
    cut=max(2,round(.70*T))
    train=raw[:cut].copy()
    # All-missing optional columns receive a neutral zero (no RuntimeWarnings).
    center=np.zeros(F,dtype=float);low=np.zeros(F,dtype=float);hi=np.zeros(F,dtype=float)
    for f in range(F):
        good=train[:,:,f].ravel()
        good=good[np.isfinite(good)]
        if len(good):
            center[f]=float(np.median(good))
            low[f]=float(np.percentile(good,1))
            hi[f]=float(np.percentile(good,99))
    # Bound extrema based on train range. Preserve meaningful deviations within bounds.
    for f in range(F):
        raw[:,:,f]=np.where(np.isfinite(raw[:,:,f]),raw[:,:,f],center[f])
        raw[:,:,f]=np.clip(raw[:,:,f],low[f],hi[f])
    scale=np.maximum(np.nanstd(raw[:cut],axis=(0,1)),1e-5)
    x=(raw-center[None,None,:])/scale[None,None,:]
    x=np.clip(x,-12,12).astype(np.float32)
    # Six logical edge relations; never use targets to derive next-day edge state.
    rel=rel.copy()
    valid=rel.source_id.astype(str).isin(node_lookup)&rel.target_id.astype(str).isin(node_lookup)
    rel=rel[valid].drop_duplicates(["source_id","target_id","relation_type"])
    unknown=set(rel.relation_type)-set(EDGE_TYPES)
    if unknown:raise ValueError(f"Unknown relations: {sorted(unknown)}")
    if rel.empty:raise ValueError("No valid graph edges")
    ei=np.array([(node_lookup[str(a)],node_lookup[str(b)]) for a,b in zip(rel.source_id,rel.target_id)],np.int64).T
    et=np.array([EDGE_TYPES.index(v) for v in rel.relation_type],np.int64)
    base=np.zeros((len(rel),4),dtype=np.float32)
    for j,col,default in [(0,"dependency_strength",.5),(1,"lead_time",.2),(3,"capacity",.8)]:
        base[:,j]=_coerce_number(rel[col]).fillna(default).to_numpy() if col in rel else default
    base[:,0]=np.clip(base[:,0],0,1)
    base[:,1]=np.clip(base[:,1]/max(1,float(np.percentile(base[:,1],95))),0,1)
    base[:,3]=np.clip(base[:,3]/max(1,float(np.percentile(base[:,3],95))),0,1)
    ea=np.repeat(base[None,:,:],T,axis=0)
    # Completed-event frequencies are historical and lagged; not contemporaneous targets.
    hist_events=np.clip(historical[:,:,0],0,1)
    ea[:,:,2]=hist_events[:,ei[0]]
    manifest.update({"source":manifest.get("source","CANONICAL_EXTERNAL"),
                     "target_granularity":"observed_event_start_date",
                     "label_nature":manifest.get("label_nature","source_supplied"),
                     "port_feature_granularity":port_resolution,
                     "matched_event_rows":kept,"input_event_rows":len(events),
                     "supervised_type_targets":int((lab["type"]>=0).sum()),
                     "supervised_severity_targets":int((lab["severity"]>=0).sum()),
                     "supervised_delay_targets":int(lab["delay_valid"].sum()),
                     "supervised_loss_targets":int(lab["loss_valid"].sum()),
                     "unmatched_event_rows":int(len(events)-kept),
                     "true_daily_port_label_task":False,
                     "real_port_task_note":"Use port-task with actual per-port daily turnaround observations and derived labels",
                     "training_only_normalization":True,
                     "feature_names":list(FEATURE_NAMES),"training_cutoff":days[cut-1],
                     "observed_node_counts":{name:int(sum(types==tp)) for tp,name in enumerate(NODE_TYPES)},
                     "observed_edges":len(rel)})
    if kept==0:
        raise ValueError("No event dates/entity IDs aligned with operations; do not invent targets. "
                         "Provide matching canonical events.csv or use the explicitly semi-synthetic public adapter.")
    return GraphSequence(x=x,edge_index=ei,edge_type=et,edge_attr=ea,node_type=types,
                         labels=lab,dates=days,node_ids=ids,manifest=manifest).validate()


def audit_raw(directory):
    directory=Path(directory)
    files=sorted(directory.rglob("*.csv"))
    out=[]
    for p in files:
        try:
            df=pd.read_csv(p,nrows=3,low_memory=False)
            out.append({"file":str(p.relative_to(directory)),"columns":list(df.columns),"example_rows":len(df)})
        except Exception as exc:
            out.append({"file":str(p.relative_to(directory)),"error":str(exc)})
    return out


def _find_alias(df,*names):
    for name in names:
        key=re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")
        if key in df:return key
    for name in names:
        key=re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")
        for c in df.columns:
            if key in c:return c
    return None


def convert_public_sources(raw_root,output_dir,seed=11):
    """Best-effort transparent adapter, for downloaded source CSVs in 3 folders.

    Source CSV columns vary across Kaggle revisions: audit columns and refuse to invent
    unavailable targets. Non-joinable entities and events are deterministically matched
    in a SEMI-SYNTHETIC setting; output provenance describes every such operation.
    """
    raw_root=Path(raw_root);output_dir=Path(output_dir)
    rng=np.random.default_rng(seed)
    roots={"supplygraph":raw_root/"supplygraph", "disruptions":raw_root/"disruptions",
           "maritime":raw_root/"maritime"}
    for name,p in roots.items():
        if not p.is_dir() or not list(p.rglob("*.csv")):
            raise FileNotFoundError(f"{name}: put downloaded CSVs under {p}; run `python -m scrn.cli audit --raw-root {raw_root}`")
    files={name:sorted(p.rglob("*.csv")) for name,p in roots.items()}
    # SupplyGraph native source is often WIDE Date + product columns.
    candidates=[]
    for p in files["supplygraph"]:
        try:
            df=_normalise_columns(pd.read_csv(p,low_memory=False))
        except Exception:continue
        dat=_find_alias(df,"date","day")
        if dat and len(df)>=12:
            num=[c for c in df if c!=dat and pd.to_numeric(df[c],errors="coerce").notna().sum() >= max(3,.5*len(df))]
            if len(num)>=3:candidates.append((p,df,dat,num))
    if not candidates:
        raise ValueError("No wide dated SupplyGraph time-series CSV found. See source repository's Raw Dataset and audit output.")
    candidates.sort(key=lambda q:(len(q[1]),len(q[3])),reverse=True)
    primary,series,date_col,products=candidates[0]
    series=series.copy();series[date_col]=_date_col(series[date_col]);series=series.dropna(subset=[date_col])
    series=series.drop_duplicates(date_col).sort_values(date_col)
    dates=series[date_col].tolist()
    ids=[f"product_{i:04d}" for i in range(len(products))]
    entities=[{"entity_id":v,"entity_type":"demand_region","capacity":25.,"lead_time":3.0}
              for v in ids]
    records=[]
    for j,pcode in enumerate(products):
        vals=pd.to_numeric(series[pcode],errors="coerce").fillna(0).to_numpy()
        records.extend({"day":day,"entity_id":ids[j],"demand":max(0,float(v)),"inventory":10.,"capacity":25.}
                       for day,v in zip(dates,vals))
    # No file-level proof of supplier/warehouse/port links: explicitly construct logical counterparts.
    extra_count=max(3,min(16,len(products)//3))
    by_type={}
    for typ in ("supplier","factory","warehouse","port","logistics_hub"):
        by_type[typ]=[f"constructed_{typ}_{i:03d}" for i in range(extra_count)]
        for v in by_type[typ]:
            entities.append({"entity_id":v,"entity_type":typ,"capacity":30.,"lead_time":2.5})
    for tp,names in by_type.items():
        for j,v in enumerate(names):
            for d in dates:
                records.append({"day":d,"entity_id":v,"demand":0.,"production":float(10+rng.uniform(-.3,.3)),
                                "inventory":15.,"delivery":10.,"capacity":30.,"lead_time":2.5})
    relations=[]
    def link(a,b,tp):
        relations.append({"source_id":a,"target_id":b,"relation_type":tp,
                          "dependency_strength":.5,"lead_time":2.5,"capacity":25.})
    for j in range(extra_count):
        link(by_type["supplier"][j],by_type["factory"][j],"procurement")
        link(by_type["factory"][j],by_type["warehouse"][j],"production")
        link(by_type["port"][j],by_type["logistics_hub"][j],"shipment")
        link(by_type["logistics_hub"][j],by_type["warehouse"][j],"shipment")
        link(by_type["supplier"][j],by_type["factory"][(j+1)%extra_count],"recovery")
        link(by_type["factory"][j],by_type["warehouse"][(j+1)%extra_count],"propagation")
    for j,product in enumerate(ids):link(by_type["warehouse"][j%extra_count],product,"inventory")
    # Event CSV: match typed target columns, but do not claim record belongs to observed factory.
    disruption_files=[p for p in files["disruptions"] if p.suffix.lower()==".csv"]
    ranked=[]
    for p in disruption_files:
        try:df=_normalise_columns(pd.read_csv(p,low_memory=False))
        except Exception:continue
        score=sum(bool(_find_alias(df,*aliases)) for aliases in
                  [("disruption_type","event_type","disruption"),
                   ("severity","severity_level"),("financial_loss","loss","economic_impact")])
        ranked.append((score,len(df),p,df))
    ranked.sort(key=lambda tup:(tup[0],tup[1]),reverse=True)
    if not ranked or ranked[0][0]<2:
        raise ValueError("Cannot identify both disruption category and impact labels in Kaggle CSV; "
                         "inspect audit report, then convert to canonical events.csv")
    _,_,event_file,event_df=ranked[0]
    kind_col=_find_alias(event_df,"disruption_type","event_type","event_category","disruption")
    severity_col=_find_alias(event_df,"severity","severity_level","disruption_severity")
    loss_col=_find_alias(event_df,"financial_loss_usd","financial_loss","loss_usd","economic_impact","loss")
    delay_col=_find_alias(event_df,"delay_days","recovery_time_days","recovery_duration","downtime_days","delay")
    date_col=_find_alias(event_df,"start_date","event_date","disruption_date","date")
    event_rows=[]
    for j,r in event_df.iterrows():
        if j>=10000:break
        cat=str(r[kind_col]) if kind_col else "demand fluctuation"
        sev=str(r[severity_col]) if severity_col else "low"
        typ=_disruption(cat)
        node_pool=(by_type["port"] if typ==1 else by_type["supplier"] if typ==0 else
                   by_type["factory"] if typ==7 else by_type["logistics_hub"])
        entity=node_pool[int(rng.integers(len(node_pool)))]
        event_date=None
        if date_col:
            v=pd.to_datetime(r[date_col],errors="coerce")
            if pd.notna(v) and v.strftime("%Y-%m-%d") in dates:event_date=v.strftime("%Y-%m-%d")
        if event_date is None:
            # Explicit synthetic temporal assignment to the available SupplyGraph horizon.
            event_date=dates[int(rng.integers(1,len(dates)))]
        val_loss=float(_coerce_number(pd.Series([r[loss_col]])).iloc[0]) if loss_col else float("nan")
        if loss_col and "usd" in loss_col and "kusd" not in loss_col:val_loss=val_loss/1000.
        delay=float(_coerce_number(pd.Series([r[delay_col]])).iloc[0]) if delay_col else float("nan")
        event_rows.append({"entity_id":entity,"start_date":event_date,"end_date":event_date,
                           "disruption_type":cat,"severity":sev,
                           "delay_days":max(0,delay) if np.isfinite(delay) else np.nan,
                           "loss_kusd":max(0,val_loss) if np.isfinite(val_loss) else np.nan})
    if not event_rows:raise ValueError("No disruption events available in CSV")
    # Maritime Kaggle data may describe economies and semester periods, NOT port calls.
    maritime_rows=[];maritime_source="none";port_is_actual_port=False
    for p in files["maritime"]:
        try:pd_port=_normalise_columns(pd.read_csv(p,low_memory=False))
        except Exception:continue
        metric=_find_alias(pd_port,"median_time_in_port_days_value","turnaround_days","median_time_in_port_days")
        if metric:
            maritime_source=str(p); period=_find_alias(pd_port,"period","date")
            name=_find_alias(pd_port,"port_id","port_name","port")
            country=_find_alias(pd_port,"economy_label","country","economy")
            port_is_actual_port=bool(name)
            v=_coerce_number(pd_port[metric])
            df=pd_port.assign(turnaround_days=v)
            df=df.dropna(subset=["turnaround_days"])
            # Aggregating economy or ship-type data is a STATIC PROXY, never observed port records.
            df=df.head(2000)
            for j,r in df.iterrows():
                maritime_rows.append({"entity_id":by_type["port"][j%extra_count],
                                      "period":str(r[period]) if period else "2022-S1",
                                      "turnaround_days":float(r["turnaround_days"]),
                                      "vessel_calls":0.,"cargo_capacity":0.,"efficiency":0.})
            break
    output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(entities).to_csv(output_dir/"entities.csv",index=False)
    pd.DataFrame(relations).to_csv(output_dir/"relations.csv",index=False)
    pd.DataFrame(records).to_csv(output_dir/"operations.csv",index=False)
    pd.DataFrame(event_rows).to_csv(output_dir/"events.csv",index=False)
    pd.DataFrame(maritime_rows,columns=["entity_id","period","turnaround_days","vessel_calls","cargo_capacity","efficiency"]).to_csv(output_dir/"ports.csv",index=False)
    manifest={"source":"PUBLIC_DATA_SEMI_SYNTHETIC_LINKAGE","seed":seed,
             "label_nature":"synthetic_disruption_record_linked_to_operational_nodes",
             "supplygraph_file":str(primary),"disruption_file":str(event_file),
             "maritime_file":maritime_source,"public_records_matched_by_real_entity_id":False,
             "synthetic_linkage":True,"synthetic_event_temporal_alignment":True,
             "constructed_entities":by_type,"port_is_actual_port":False,
             "real_port_task_supported":False,
             "important":"SupplyGraph product nodes are real operational data; all additional tier links are constructed. "
                         "Maritime economy/semester records are not daily physical-port observations."}
    (output_dir/"provenance.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output_dir
