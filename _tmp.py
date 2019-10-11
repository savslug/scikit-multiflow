
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.trees.perfect_random_tree import PerfectRandomTree
from skmultiflow.trees.hoeffding_tree import HoeffdingTree

HT=HoeffdingTree()
PRT=PerfectRandomTree()

stream=SEAGenerator()
stream.prepare_for_use()
print(stream.n_classes)
evaluator = EvaluateHoldout(max_samples=400000,
                            max_time=1000,
                            n_wait=1000,
                            show_plot=True,
                            metrics=['accuracy','running_time','model_size'],
                            dynamic_test_set=True)

# Run evaluation
#evaluator.evaluate(stream=stream, model=[ht_bag,hat_bag, hatt_bag], model_names=['HT','HAT',"HATT"])
evaluator.evaluate(stream=stream, model=[HT,PRT], model_names=["HT","PRT"])
#ARS.partial_fit([[1],[2],[3]],[1,2,3],classes=3)
#ARS.predict([[1,2,3]])