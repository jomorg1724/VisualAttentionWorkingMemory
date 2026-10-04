"""Build only this architecture paper; never execute model/training code."""
from pathlib import Path
import os, json, subprocess, shutil
HERE = Path(__file__).resolve().parent
counts = json.loads((HERE/'count_verification.json').read_text())
(HERE/'model_hash.tex').write_text(counts['source_sha256']['SecondPass/SpatialReadout/model.py'])
engine = os.environ.get('TECTONIC') or shutil.which('tectonic') or '/Users/jonathanmorgan/.local/bin/tectonic'
result = subprocess.run([engine,'--keep-logs','--keep-intermediates','SpatialReadout_Architecture.tex'],cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(HERE/'build_output.txt').write_text(result.stdout)
print(result.stdout)
raise SystemExit(result.returncode)
