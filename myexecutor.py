from pddlsim.local_simulator import LocalSimulator
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
import sys
from mcts import monte_carlo_tree_search, init_helper_objects, sample_actions
MAX_SIMULATIONS = 50


class MyExecutor(Executor):
    """
    Executor which uses Monte Carlo Tree Search algorithm.
    """
    def __init__(self, min_tries, max_simulation_depth, action_max_tries):
        super(MyExecutor, self).__init__()
        self.services = None
        self._prev_state = None
        self._prev_action = None
        self._min_tries = min_tries
        self._max_simulation_depth = max_simulation_depth
        self._action_max_tries = action_max_tries
        self._action_retry_counter = 0

    def initialize(self, services):
        self.services = services

        # Set the number of simulations.
        max_tries = min(MAX_SIMULATIONS, max(self._min_tries, len(self.services.valid_actions.get()) * 2))

        # Initialize helper objects for the MCTS algorithm.
        init_helper_objects(services, max_tries, self._max_simulation_depth, self._action_max_tries)

    def next_action(self):

        # Check if reached all goals.
        if self.services.goal_tracking.reached_all_goals():
            return None

        # Get all the valid actions.
        valid_actions = sample_actions(self.services.valid_actions.get())

        # Check if there are no valid actions to take.
        if len(valid_actions) == 0:
            return None

        # Get the current state.
        curr_state = self.services.perception.get_state()

        # Check if there is only one valid action.
        if len(valid_actions) == 1:
            self._prev_state = curr_state
            self._prev_action = valid_actions[0]
            return valid_actions[0]

        # Check if the previous action failed.
        if curr_state == self._prev_state:

            # Check if the action can be retried.
            if self._action_retry_counter < self._action_max_tries:
                self._action_retry_counter += 1
                return self._prev_action
            else:
                # If an action failed more than the limit, remove it from the valid actions list.
                valid_actions.remove(self._prev_action)

                # Check if only one valid action was left.
                if len(valid_actions) == 1:
                    self._prev_action = valid_actions[0]
                    return valid_actions[0]

        # Use the MCTS algorithm to choose an action.
        action = monte_carlo_tree_search(curr_state, valid_actions, self._prev_state)

        self._prev_state = curr_state
        self._prev_action = action
        self._action_retry_counter = 0

        return action


if __name__ == '__main__':

    domain_path = sys.argv[1]
    problem_path = sys.argv[2]

    # Initializing parameters.
    min_tries = 3
    max_simulation_depth = 3
    action_max_tries = 2

    executor = MyExecutor(min_tries, max_simulation_depth, action_max_tries)

    print LocalSimulator(local).run(domain_path, problem_path, executor)
