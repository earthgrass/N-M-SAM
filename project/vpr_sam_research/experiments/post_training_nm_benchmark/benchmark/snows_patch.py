"""Apply a reviewable patch IN MEMORY; never edit the pinned official checkout."""
import difflib,inspect,sys,types
from pathlib import Path
from . import snows_repair_kernels as kernels
from .upstream import ROOT,source

def candidate_source():
    original=source('SNOWS','CNN_pruning/prune/SNOWS.py');text=original
    names=['conjugate_gradient_sparse','hessian_vector_product_chunks']
    for name in names:
        start=text.index('def '+name+'(');end=text.find('\ndef ',start+1)
        if end<0:raise ValueError('Unexpected upstream function boundary')
        replacement=inspect.getsource(getattr(kernels,name))+'\n'
        text=text[:start]+replacement+text[end:]
    # The unused trailing placeholder has no statement and prevents parsing.
    # Never pretend to implement it: calling it must explicitly fail.
    text+='\n    raise NotImplementedError("Unfinished upstream prune_and_update; not used/approved")\n'
    patch=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
        fromfile='a/CNN_pruning/prune/SNOWS.py',tofile='b/CNN_pruning/prune/SNOWS.py'))
    return text,patch

def load_candidate():
    text,patch=candidate_source()
    if (ROOT/'patches/snows_minimal_diagnostic.patch').read_text(encoding='utf-8')!=patch:
        raise ValueError('Controlled SNOWS patch differs from reviewed diff')
    helpers=types.ModuleType('_snows_controlled_helpers')
    exec(compile(source('SNOWS','CNN_pruning/prune/helpers.py'),'controlled/SNOWS/helpers.py','exec'),helpers.__dict__)
    saved=sys.modules.get('helpers');sys.modules['helpers']=helpers
    module=types.ModuleType('_snows_controlled_candidate')
    try:exec(compile(text,'controlled/SNOWS/SNOWS.py','exec'),module.__dict__)
    finally:
        if saved is None:sys.modules.pop('helpers',None)
        else:sys.modules['helpers']=saved
    return module
