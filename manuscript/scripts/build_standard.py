"""Build the separate single-column Times New Roman reading edition."""
from pathlib import Path
import subprocess,sys,hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
commands=[['xelatex','-interaction=nonstopmode','-halt-on-error','main_standard.tex'],
          ['bibtex','main_standard'],
          ['xelatex','-interaction=nonstopmode','-halt-on-error','main_standard.tex'],
          ['xelatex','-interaction=nonstopmode','-halt-on-error','main_standard.tex']]
for i,cmd in enumerate(commands):
    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'review'/f'standard_build_pass_{i+1}.log').write_bytes(p.stdout)
    if p.returncode:
        print(p.stdout.decode(errors='replace')[-5000:]);sys.exit(p.returncode)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[ROOT/'main_standard.tex',ROOT/'references.bib',*sorted((ROOT/'sections').glob('*.tex')),*sorted((ROOT/'tables').glob('*.tex'))]
record={'status':'BUILT','command':'python scripts/build_standard.py','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pdf':'main_standard.pdf','pdf_sha256':sha(ROOT/'main_standard.pdf'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'format':'A4; single column; 10pt Times New Roman; 23mm margins; Roman main headings; A/B/C subheadings; XeLaTeX; unsrtnat references'}
(ROOT/'review/standard_build.json').write_text(json.dumps(record,indent=2)+'\n')
print('Built main_standard.pdf; preserved main.tex / main.pdf CVPR edition.')
