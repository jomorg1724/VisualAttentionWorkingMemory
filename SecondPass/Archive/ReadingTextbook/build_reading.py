"""Build the reading and run CPU-only document checks."""
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,'draw_diagrams.py'],cwd=HERE,check=True)
subprocess.run(['pandoc','visual_memory_textbook.md','--standalone','--include-in-header=header.tex','--pdf-engine=tectonic','-o','output/pdf/visual_memory_textbook.pdf'],cwd=HERE,check=True)
subprocess.run([sys.executable,'verify_reading.py'],cwd=HERE,check=True)
