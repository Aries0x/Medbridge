import time
from huggingface_hub import HfApi
api = HfApi()
for i in range(12):
    info = api.space_info("Nermal007/medbridge")
    stage = info.runtime.stage if info.runtime else "unknown"
    print(f"t={i*15}s: {stage}", flush=True)
    if stage == "RUNNING":
        print("Space is RUNNING!")
        break
    if stage in ("BUILD_ERROR", "RUNTIME_ERROR"):
        print(f"ERROR: {stage}")
        break
    time.sleep(15)
