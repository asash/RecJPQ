from aprec.recommenders.sequential.target_builders.target_builders import TargetBuilder

#returns empty targets
#can be used for self-supervised models like gpt 

class DummyTargetBuilder(TargetBuilder):
    def get_targets(self, start, end):
        return [], [0] * (end-start) #0 is a dummy value, shouldn't be used in the model

    def build(self, user_targets):
        pass
        