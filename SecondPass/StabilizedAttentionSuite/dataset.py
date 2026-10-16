"""Full unmodified13-task/35-cell suite, with fresh native stream namespaces."""
from SecondPass.TaskSuite.suite import SuiteStream,TASKS,CATALOG,task_classes

NAMESPACES=dict(train=199101,val=199102,test=199103)

class Stream(SuiteStream):
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return NAMESPACES[self.split]*100000+TASKS[task]['stream_id']*1000+index

def all_cells():return [(task,c['id']) for task,spec in TASKS.items() for c in spec['conditions']]
