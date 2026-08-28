import sys
import time
import json
from urllib import request

def get_status(job_id: str) -> dict:
    url = f"http://127.0.0.1:8000/training_status/{job_id}"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_nodes() -> dict:
    url = "http://127.0.0.1:8000/nodes"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8")).get("nodes", {})

if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else "b8047ffc-5685-4aa7-bfc4-c880c51005cc"
    
    print(f"Monitoring training job: {job_id}\n")
    start_time = time.time()
    
    while True:
        try:
            status = get_status(job_id)
            progress = status.get("progress", {})
            nodes = get_nodes()
            
            elapsed = int(time.time() - start_time)
            
            print(f"\n[{elapsed}s] Status: {status['status']}")
            print(f"Round: {progress['round']}/{progress['total_rounds']} | Done: {progress['done']}/{progress['total']} | Assigned: {progress['assigned']}")
            
            print("\nWorker Status:")
            for node_id, node in nodes.items():
                gpu_name = node.get("gpu", "").split()[-1] if node.get("gpu") else "unknown"
                print(f"  {node.get('display_name') or node_id[:8]}: {gpu_name} | Load: {node.get('load', 0):.1f} | Active: {node.get('active_batches', 0)} | Completed: {node.get('completed_batches', 0)}")
            
            if status['status'] == "done":
                print("\n✓ Training completed!")
                print(f"Total time: {elapsed}s")
                break
            
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
