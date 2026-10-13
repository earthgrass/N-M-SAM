import torch


class StatefulBatchSampler:
    """Deterministic epoch order/cursor, intended for num_workers=0 exact resume."""
    def __init__(self,size,batch_size,seed=0):
        if size<1 or batch_size<1:raise ValueError("positive sampler sizes required")
        self.size,self.batch_size,self.seed=size,batch_size,seed
        self.epoch,self.cursor=0,0
        self.order=self._order()

    def _order(self):
        return torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch))

    def __iter__(self):
        while self.cursor<self.size:
            end=min(self.cursor+self.batch_size,self.size)
            batch=self.order[self.cursor:end].tolist();self.cursor=end
            yield batch

    def next_epoch(self):
        if self.cursor!=self.size:raise ValueError("current epoch not exhausted")
        self.epoch+=1;self.cursor=0;self.order=self._order()

    def state_dict(self):
        return {k:getattr(self,k) for k in ("size","batch_size","seed","epoch","cursor","order")}

    def load_state_dict(self,state):
        for key in ("size","batch_size","seed"):
            if state[key]!=getattr(self,key):raise ValueError("sampler configuration mismatch")
        if not 0<=state["cursor"]<=self.size or sorted(state["order"].tolist())!=list(range(self.size)):
            raise ValueError("invalid sampler state")
        self.epoch,self.cursor,self.order=state["epoch"],state["cursor"],state["order"].clone()
