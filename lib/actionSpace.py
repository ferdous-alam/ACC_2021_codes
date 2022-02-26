def actionSpace():
    """
    Eight possible actions: the agent can move in these eight potential
    directions.
    move forward, move backward, move left, move right, move north-east,
    move south-east, move north-west, move south-west
    Sometimes these actions will be referred as the "primitive actions".
    """
    ACTION_FORWARD = 0
    ACTION_BACKWARD = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NE = 4
    ACTION_SE = 5
    ACTION_NW = 6
    ACTION_SW = 7

    actions = [ACTION_FORWARD, ACTION_BACKWARD,
               ACTION_LEFT, ACTION_RIGHT, ACTION_NE,
               ACTION_SE, ACTION_NW, ACTION_SW]
    n_actions = len(actions)

    return actions, n_actions
