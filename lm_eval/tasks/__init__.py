from pprint import pprint
from typing import List, Union

import lm_eval.base
import sacrebleu

from . import kobest

########################################
# All tasks : Kobest
########################################

# task 레지스트리에 클래스 추가
TASK_REGISTRY = {
    "kobest_boolq": kobest.BoolQ,
    "kobest_copa": kobest.COPA,
    "kobest_wic": kobest.WiC,
    "kobest_hellaswag": kobest.HellaSwag,
    "kobest_sentineg": kobest.SentiNeg,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
