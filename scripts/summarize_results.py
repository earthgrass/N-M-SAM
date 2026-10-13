import argparse
import json
from nmsam.experiment.compare import summarize_directory


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input-dir",default="results/formal")
    p.add_argument("--output",default="results/REPORT_RESULTS.md");p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if not args.execute:print(json.dumps({"dry_run":True,"input":args.input_dir,"output":args.output}));raise SystemExit(0)
    print(json.dumps({"artifacts":summarize_directory(args.input_dir,args.output),"report":args.output}))
