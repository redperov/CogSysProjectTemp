from pddlsim.local_simulator import LocalSimulator
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
import sys
from mcts import monte_carlo_tree_search, init_helper_objects


class MyExecutor(Executor):
    """
    My Executor.
    """
    def __init__(self, simulations_max_tries, max_simulation_depth, action_max_tries):
        super(MyExecutor, self).__init__()
        self.services = None
        self._prev_state = None
        self._prev_action = None
        self._simulations_max_tries = simulations_max_tries
        self._max_simulation_depth = max_simulation_depth
        self._action_max_tries = action_max_tries
        self._action_retry_counter = 0

        # TODO maybe change to a numpy array to increase the speed.
        # self._black_list = []

    def initialize(self, services):
        self.services = services

        # Initialize helper objects for the MCTS algorithm.
        init_helper_objects(services, simulations_max_tries, max_simulation_depth, action_max_tries)

    def next_action(self):

        # Check if reached all goals.
        if self.services.goal_tracking.reached_all_goals():
            return None

        # Get all the valid actions.
        valid_actions = self.services.valid_actions.get()

        # Check if there are no valid actions to take.
        if len(valid_actions) == 0:
            return None

        # Get the current state.
        curr_state = self.services.perception.get_state()

        # # Check if the current state already appears in the black list.
        # if curr_state not in self._black_list:
        #
        #     # Add the current state to the black list.
        #     self._black_list.append(curr_state)

        # Check if there is only one valid action.
        if len(valid_actions) == 1:
            self._prev_state = curr_state
            return valid_actions[0]

        # Check if the previous action failed, if it did, check if it can be retried.
        if curr_state == self._prev_state and self._action_retry_counter < self._action_max_tries:
            self._action_retry_counter += 1
            return self._prev_action

        # Use the MCTS algorithm to choose an action.
        # TODO looks like the UCT encounters division by zero, fix it
        action = monte_carlo_tree_search(curr_state, valid_actions, self._prev_state)

        self._prev_action = action
        self._action_retry_counter = 0
        self._prev_state = curr_state

        return action


if __name__ == '__main__':

    domain_path = sys.argv[1]
    problem_path = sys.argv[2]

    # Initializing parameters.
    simulations_max_tries = 3
    max_simulation_depth = 3
    action_max_tries = 2

    executor = MyExecutor(simulations_max_tries, max_simulation_depth, action_max_tries)

    print LocalSimulator(local).run(domain_path, problem_path, executor)
