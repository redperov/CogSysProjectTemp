"""
Implementation of the Monte Carlo Tree Search algorithm.
"""
import numpy as np
from valid_action import PythonValidActions
from random import choice

NUM_OF_SIMULATIONS = 3
C = 1.4
MAX_ROLLOUT_DEPTH = 10

# Rewards values
DEAD_END_REWARD = -1
ACTIONS_LEFT_REWARD = 0
GOAL_COMPLETED_REWARD = 10

valid_actions_getter = None
sim_services = None
sim_simulations_max_tries = 0
sim_max_simulation_depth = 0
sim_action_max_tries = 0
sim_prev_state = None
# sim_black_list = None


# class State(object):
#
#     def __init__(self, str_state):
#         """
#         Constructor.
#         :param str_state: string pddl representation of the state.
#         """
#         self._str_state = str_state


class Node(object):

    def __init__(self, str_state, parent, valid_actions=None, applied_action=None):
        """
        Constructor.
        :param str_state: string pddl representation of the state.
        :param parent: parent state
        """
        self._state = str_state
        self._parent = parent
        self._children = None
        self._valid_actions = valid_actions
        self._applied_action = applied_action
        self._visit_count = 0
        self._win_score = 0

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_children(self):

        # Check if the current state tried to create any children.
        if self._children is None:
            self._children = []

            # Create a child for every available action.
            for action in self.get_valid_actions():

                # The state on which the action will be applied.
                state = sim_services.parser.copy_state(self._state)

                # Used for checking whether the action succeeded.
                prev_state = sim_services.parser.copy_state(self._state)

                # Counts the number of times an action failed.
                counter = 0

                # TODO maybe there is a need for try except here
                while counter < sim_action_max_tries:

                    # Apply the action on the state.
                    sim_services.parser.apply_action_to_state(action, state)
                    counter += 1

                    # Check if the state has changed.
                    if state != prev_state:

                        # Check if the state is not in the black list.
                        if state != sim_prev_state:
                            # Add the new state to the list.
                            self._children.append(Node(state, self, applied_action=action))

                        break

        return self._children

    def get_visit_count(self):
        return self._visit_count

    def increase_visit_count(self):
        self._visit_count += 1

    def add_win_score(self, score):
        self._win_score += score

    def get_win_score(self):
        return self._win_score

    # def get_all_possible_states(self):
    #     pass

    def get_valid_actions(self):

        if self._valid_actions is None:
            self._valid_actions = valid_actions_getter.get(self._state)

        return self._valid_actions

    def get_applied_action(self):
        return self._applied_action


def init_helper_objects(services, simulations_max_tries, max_simulation_depth, action_max_tries):
    global valid_actions_getter, sim_services, sim_simulations_max_tries, sim_max_simulation_depth, sim_action_max_tries
    sim_services = services
    valid_actions_getter = PythonValidActions(sim_services.parser, sim_services.perception)
    sim_simulations_max_tries = simulations_max_tries
    sim_max_simulation_depth = max_simulation_depth
    sim_action_max_tries = action_max_tries
    #sim_black_list = black_list


def monte_carlo_tree_search(pddl_state, valid_actions, prev_state):

    global sim_prev_state
    sim_prev_state = prev_state

    root = Node(pddl_state, None, valid_actions=valid_actions)

    for i in xrange(NUM_OF_SIMULATIONS):
        leaf = traverse(root)  # leaf = unvisited node
        simulation_result = rollout(leaf)
        back_propagate(leaf, simulation_result)

    # Get the best action.
    action = best_action(root)

    return action


def traverse(node):
    depth = 0

    while fully_expanded(node) and depth < sim_max_simulation_depth:
        node = best_uct(node)
        depth += 1

    return pick_unvisited(node.get_children()) or node  # in case no children are present / node is terminal


def fully_expanded(node):
    """
    Checks if all the node's children are visited.
    :param node: node
    :return: boolean
    """
    children = node.get_children()

    if len(children) == 0:
        return False

    for child in children:
        if child.get_visit_count() == 0:
            return False

    return True


def best_uct(node):

    # If there is a child with a num of visits which is 0, its UCT equals infinity, so it's the best child to choose.
    for c in node.get_children():
        if c.get_visit_count == 0:
            return c

    # TODO maybe avoid doing two separate loops, and combine them to one for better performance speed
    choices_weights = [
        (c.get_win_score() / (c.get_visit_count())) + C * np.sqrt((2 * np.log(node.get_visit_count()) /
                                                                   (c.get_visit_count())))
        for c in node.get_children()]

    return node.get_children()[np.argmax(choices_weights)]


def pick_unvisited(children):
    # TODO if using numpy array, change to .size()
    if len(children) == 0:
        return None

    for child in children:
        if child.get_visit_count() == 0:
            return child

    return None


def rollout(node):
    curr_depth = 0

    while not is_terminal(node) and curr_depth < MAX_ROLLOUT_DEPTH:
        node = rollout_policy(node)
        curr_depth += 1

    return get_result(node)


def is_terminal(node):
    # Check if reached one of the goals.
    if is_reached_a_goal_state(node.get_state()):
        return True

    # Check if there aren't any child states to move to.
    if len(node.get_children()) == 0:
        return True

    return False


def is_reached_a_goal_state(state):
    # Get all the uncompleted goals.
    goals = sim_services.goal_tracking.uncompleted_goals

    for goal in goals:

        # Test the state to see if it matches any of the goal states.
        result = goal.test(state)

        # Check if a goal was completed.
        if result:
            return True

    return False


def rollout_policy(node):
    return choice(node.get_children())


def get_result(node):
    # TODO maybe if stopped because reached max depth return x and if got to dead end return y?

    # Check if reached one of the goals.
    if is_reached_a_goal_state(node.get_state()):
        return GOAL_COMPLETED_REWARD

    # Check if reached a dead end.
    if len(node.get_valid_actions()):
        return DEAD_END_REWARD

    return ACTIONS_LEFT_REWARD


def back_propagate(node, result):

    # Update statistics.
    node.increase_visit_count()
    node.add_win_score(result)

    # Check if the node is not a root.
    if node.get_parent():

        # Continue back propagating.
        # TODO maybe avoid doing recursion.
        back_propagate(node.get_parent(), result)

    # # Check if node is root.
    # if node.get_parent() is None:
    #     return
    #
    # # Update node's statistics.
    # update_stats(node, result)
    #
    # # Continue back propagating.
    # back_propagate(node.get_parent(), result)


# def update_stats(node, result):
#     node.increase_visit_count()
#     node.add_win_score(result)


def best_action(node):

    children = node.get_children()

    # TODO check if it's possible for the root to not have any children
    if len(children) == 0:
        return None

    # Pick the child with the highest number of visits.
    visit_count_list = [child.get_visit_count() for child in node.get_children()]
    child = node.get_children()[np.argmax(visit_count_list)]

    # Return the action which brought to him.
    return child.get_applied_action()
