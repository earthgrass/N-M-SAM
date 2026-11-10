"""Pinned official source access. No latest-branch fallback or formula substitution."""
import ast,hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
import numpy as np
import torch
from .common import sha_file

ROOT=Path(__file__).resolve().parents[1]

def source(repo,name,root=None):
    lock=json.loads((ROOT/'configs/upstream_lock.json').read_text(encoding='utf-8'))
    folder=Path(root) if root is not None else ROOT/'artifacts/upstream'/repo
    commit=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    if commit!=lock[repo]['commit']:raise ValueError('Wrong official commit: '+repo)
    expected=lock[repo]['files'][name]['sha256']
    try:path=folder/name;present=path.is_file()
    except OSError:present=False
    data=path.read_bytes() if present else subprocess.check_output(['git','-C',str(folder),'show',commit+':'+name])
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Official source bytes changed: '+repo+'/'+name)
    return data.decode('utf-8')

def selected_definitions(repo,name,names,root=None,extra=None,diagnostic_allow_invalid_module=False):
    """Execute reviewed original AST definitions unchanged, with explicit dependencies.

    Used for isolated CPU diagnostics, not as a substitute full implementation.
    """
    text=source(repo,name,root);syntax_error=None
    try:tree=ast.parse(text)
    except SyntaxError as exc:
        if not diagnostic_allow_invalid_module:raise
        # Diagnostic only: extract untouched complete top-level definitions.
        # Never add a pass, patch a body, or treat this as a working full module.
        lines=text.splitlines(keepends=True);starts=[i for i,line in enumerate(lines) if line.startswith(('def ','class '))]
        starts.append(len(lines));nodes=[]
        for start,end in zip(starts,starts[1:]):
            identifier=lines[start].split()[1].split('(')[0].split(':')[0]
            if identifier in names:nodes.extend(ast.parse(''.join(lines[start:end])).body)
        tree=ast.Module(body=nodes,type_ignores=[]);syntax_error={'line':exc.lineno,'message':exc.msg}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing official definition')
    namespace={'torch':torch,'np':np,'nn':torch.nn,'Variable':torch.autograd.Variable}
    if extra:namespace.update(extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),repo+'/'+name,'exec'),namespace)
    namespace['_whole_module_syntax_error']=syntax_error
    return namespace
