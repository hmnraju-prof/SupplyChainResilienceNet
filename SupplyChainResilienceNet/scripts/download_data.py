"""Optional downloader. Dataset licences/access controls are respected; no data bundled.

SupplyGraph is cloned via Git. Kaggle datasets use the user's own Kaggle API token.
"""
import argparse,subprocess,sys
from pathlib import Path

DATASETS={
    "disruptions":"likithagedipudi/supply-chain-disruption-and-recovery-dataset",
    "maritime":"jeleeladekunlefijabi/maritime-port-performance-dataset"
}

def run(args):
    root=Path(args.directory);root.mkdir(parents=True,exist_ok=True)
    names=["supplygraph","disruptions","maritime"] if args.dataset=="all" else [args.dataset]
    for name in names:
        target=root/name;target.mkdir(parents=True,exist_ok=True)
        if name=="supplygraph":
            dest=target/"SupplyGraph"
            if (dest/".git").exists():
                print("SupplyGraph repository already present:",dest)
            else:
                subprocess.run(["git","clone","--depth","1","https://github.com/ciol-researchlab/SupplyGraph.git",str(dest)],check=True)
        else:
            try:import kagglehub
            except ImportError:sys.exit("Install Kaggle client: pip install kagglehub; authenticate if requested")
            import shutil
            location=Path(kagglehub.dataset_download(DATASETS[name]))
            for p in location.rglob("*.csv"):
                shutil.copy2(p,target/p.name)
            print(f"Downloaded {name} to {target}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["all","supplygraph","disruptions","maritime"],default="all")
    p.add_argument("--directory",default="data/raw")
    run(p.parse_args())
