import os
from pathlib import Path

path = Path(__file__).parent
scripts = sorted(list(path.glob("[0-9]_*.py")))
for script in scripts:
    print(script)
    os.system(f'python {script}')