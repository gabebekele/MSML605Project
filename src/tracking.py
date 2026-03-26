import os
import json
from datetime import datetime


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_run_record(run_id, split_name, pair_version, threshold_info, metrics, note="", parameters=None):
    
    import datetime

    record = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "split": split_name,
        "pair_version": pair_version,
        "threshold_info": threshold_info,
        "metrics": metrics,
        "parameters": parameters or {},
        "note": note
    }
    return record


def save_run(outputs_dir, run_id, run_record):
    run_dir = os.path.join(outputs_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    output_path = os.path.join(run_dir, "run.json")

    with open(output_path, "w") as f:
        json.dump(run_record, f, indent=2)

    return output_path
