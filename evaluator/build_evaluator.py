from evaluator.top_down_human_pose import TopDownHumanPoseEval

EVALUATORS = {
    "TopDownHumanPoseEval": TopDownHumanPoseEval
}


def build_evaluator(cfg, model):
    return EVALUATORS[cfg.name](cfg, model)
