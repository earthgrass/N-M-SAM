from itertools import product
import torch
from nmsam.sparsity.mask import enumerate_masks
from nmsam.sparsity.layout import MaskLayout


def block_candidates(context,name,block_id=0):
    if name not in context.sparse_names:raise ValueError("candidate outside sparse scope")
    MaskLayout(tuple(context.shapes[context.names.index(name)]))
    sl=context.parameter_slice(name)
    width=sl.stop-sl.start
    if width%4 or block_id<0 or 4*(block_id+1)>width:
        raise ValueError("candidate block outside complete 2:4 weight")
    start=sl.start+4*block_id
    for index,mask in enumerate(enumerate_masks(device=context.z.device)):
        support=context.support.clone();support[start:start+4]=mask
        q=(support.to(context.z.dtype)-context.support.to(context.z.dtype))*context.master
        yield {"candidate_id":index,"support":support,"q":q}


def joint_candidates(context,blocks,*,max_combinations=36):
    if len(set(blocks))!=len(blocks) or not blocks:raise ValueError("distinct blocks required")
    if 6**len(blocks)>max_combinations:raise ValueError("joint candidate budget exceeded")
    # Explicit enumeration budget; do not materialize all full-size vectors.
    for ids in product(range(6),repeat=len(blocks)):
        support=context.support.clone()
        for (name,block),index in zip(blocks,ids):
            if name not in context.sparse_names:raise ValueError("joint candidate outside sparse scope")
            MaskLayout(tuple(context.shapes[context.names.index(name)]))
            sl=context.parameter_slice(name);start=sl.start+4*block
            if start+4>sl.stop or block<0:raise ValueError("block outside layout")
            support[start:start+4]=enumerate_masks(device=support.device)[index]
        q=(support.to(context.z.dtype)-context.support.to(context.z.dtype))*context.master
        yield {"candidate_id":ids,"support":support,"q":q}
