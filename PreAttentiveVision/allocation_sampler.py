"""Versioned task-local fresh streams for the two training allocations."""
from PreAttentiveVision.neuroscience_stimuli import TaskStream,TASK_CLASSES

TASK_ORDER=tuple(TASK_CLASSES)
SCHEDULES={
    'uniform':TASK_ORDER,
    'contour_focus':tuple(t for other in TASK_ORDER if other!='contour' for t in ('contour',other)),
}
TRAIN_BASE_SEED=270001


class TaskLocalStreams:
    def __init__(self,seed,split):
        self.seed=int(seed);self.split=split
        self.derived_seeds={task:self.seed+100003*(i+1) for i,task in enumerate(TASK_ORDER)}
        self.streams={task:TaskStream(value,split=split) for task,value in self.derived_seeds.items()}

    def batch(self,n,task):return self.streams[task].batch(n,task=task)

    def state_dict(self):
        return dict(version='task_local_allocation_v1',base_seed=self.seed,split=self.split,
            derived_seeds=self.derived_seeds,streams={t:s.state_dict() for t,s in self.streams.items()})

    def load_state_dict(self,state):
        if (state['version'],state['base_seed'],state['split'],state['derived_seeds'])!=('task_local_allocation_v1',self.seed,self.split,self.derived_seeds):
            raise ValueError('Task-local sampler identity mismatch')
        for task,stream in self.streams.items():stream.load_state_dict(state['streams'][task])


def exposure_counts(updates,arm):
    schedule=SCHEDULES[arm]
    return {t:sum(schedule[i%len(schedule)]==t for i in range(updates)) for t in TASK_ORDER}
