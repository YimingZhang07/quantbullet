import os, sys
print("exe:", sys.executable)
print("ver:", sys.version)
print("cwd:", os.getcwd())
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("PYTHONHOME:", os.environ.get("PYTHONHOME"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("PATH head:", os.environ.get("PATH","")[:200])
print("\n--- sys.path (top 30) ---")
for p in sys.path[:30]:
    print(p)
