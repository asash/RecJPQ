import random
from aprec.recommenders.sequential.targetsplitters.targetsplitter import TargetSplitter


class IdSplitter(TargetSplitter):
    def __init__(self) -> None:
        super().__init__()
    
    def split(self, sequence):
        if len(sequence) <= self.seqence_len:
            return sequence, []
        else:
            start = random.randint(0, len(sequence) - self.seqence_len - 1)
            end = start + self.seqence_len
            return sequence[start:end], [] #0 is a dummy value which we won't use