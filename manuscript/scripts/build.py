"""Build the actual CVPR draft and retain each compiler pass log."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
          ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
for i,cmd in enumerate(commands):
    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'review'/f'build_pass_{i+1}.log').write_bytes(p.stdout)
    if p.returncode:
        print(p.stdout.decode(errors='replace')[-6000:]);sys.exit(p.returncode)
print('Built main.pdf using pdflatex / bibtex / pdflatex / pdflatex')
