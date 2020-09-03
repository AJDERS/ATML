
import random
import configparser
import numpy as np
from tqdm import tqdm


class Config():
 
    def __init__(self, config_path):
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        # ENVIRONMENT
        cfg_env = cfg['ENVIRONMENT']
        self.rows = cfg_env.getint('Rows')
        self.cols = cfg_env.getint('Columns')
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.obstacles = self._load_states(
            cfg_env.get('ObstacleStates'), single=False
        )
        self.severe_obstacles = self._load_states(
            cfg_env.get('SevereObstacleStates'), single=False
        )
        self.start_state = self._load_states(cfg_env.get('StartState'))
        self.win_state = self._load_states(cfg_env.get('TerminalState'))

        # REWARDS
        cfg_rew = cfg['REWARDS']
        self.action_reward = cfg_rew.getint('ActionReward')
        self.severe_obstacles_reward = cfg_rew.getint('SevereObstacleReward')
        self.obstacles_reward = cfg_rew.getint('ObstacleReward')
        self.win_reward = cfg_rew.getint('WinningReward')

        # ALGORITHM
        cfg_alg = cfg['ALGORITHM']
        self.discount_rate = cfg_alg.getfloat('DiscountRate')
        self.stepsize = cfg_alg.getfloat('StepSize')
        self.exploration_rate = cfg_alg.getfloat('ExplorationRate')


    def _load_states(self, string, single=True):
        """
        Simple helper function to parse coordinate input from config

        Parameters: 
            string (str): coordinate input as a string, delimited by `\n`.
          
        Returns: 
            loaded_states (list): list of tuples containing coordinates.
            single (bool): if True a single coordinate is assumed.
        """
        coordinates = string.split('\n')
        loaded_states = []
        for coord in coordinates:
            coord_tuple = tuple(map(lambda x: int(x), coord.split(',')))
            loaded_states.append(coord_tuple)
        if single:
            return loaded_states[0]
        else:
            return loaded_states

class PolicyMaker():
    """ 
    This class defines the policy.
      
    Attributes: 
        config (Config): The configurations passed to the Config class.
    """
    def __init__(self, config):
        self.config = config

    def update_policy(
        self,
        cur_state,
        new_reward,
        new_state,
        policy=None
    ):
        """
        Updates policy based on given action, using Temporal Difference.

        Parameters: 
            cur_state (tuple): The current state.
            new_reward (int): The reward obtained from taking the next move.
            new_state (tuple): The state after taking the next move.
            policy (np.array): The state value look-up matrix.
          
        Returns: 
            policy (np.array): The updated state value look-up matrix.
        """
        try:
            len(policy)
        except TypeError:
            policy = self.config.grid

        v_t = policy[cur_state]
        alpha = self.config.stepsize
        r_t1 = new_reward
        gamma = self.config.discount_rate
        v_t1 = policy[new_state]

        policy[cur_state] = v_t + alpha * (r_t1 + gamma * v_t1 - v_t) 
        return policy

class Simulation():
    """ 
    This class defines the reinforcement learning agent and simulates its
    traversal of the environment.
      
    Attributes: 
        config_path (str): Path to config containing State, reward and agent
        parameters.
    """
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.actions = ["up", "down", "left", "right"]
        self.Environment = Environment(self.config)
        self.PolicyMaker = PolicyMaker(self.config)
        self.policy = self.config.grid

        # Define storage variables for later referencing.
        self.games = []
        self.states = []

    def choose_action(self):
        """ 
        Picks the action with highest reward according the `policy` matrix.
        """
        _next_reward = float('-inf')
        action = ""
        random.shuffle(self.actions)
        if random.uniform(0, 1) < self.config.exploration_rate:
            return self.actions[0]

        for action_ in self.actions:
            next_reward = self.policy[self.Environment.next_position(action_)]
            if next_reward >= _next_reward:
                action = action_
                _next_reward = next_reward
        return action

    def take_action(self, action):
        """
        Updates state based on given action.

        Parameters: 
            action (str): Specified action as a string.
          
        Returns: 
            self.State (State): Updated state object.
        """
        position = self.Environment.next_position(action)
        self.Environment.state = position
        return self.Environment

    def reset(self):
        """
        Resets the agent back at the starting position.
        """
        self.states = []
        self.Environment.current_reward = 0.0
        self.Environment.state = self.Environment.start_state
        self.Environment.episode_over = False

    def train(self, episodes=10):
        """
        Lets the `Agent` go through `episodes` amount of iterations of the
        Markow Decision Process.

        Parameters: 
            episodes (int): Number of episodes to complete.
        """
        i = 0
        pbar = tqdm(total = episodes)
        while i < episodes:
            if self.Environment.episode_over:
                # Append game, for plotting
                self.games.append(self.states)

                # Reset and go to next episode.
                self.reset()

                # Increment episodes
                i += 1
                pbar.update(1)
            else:
                # Store current state
                cur_state = self.Environment.state

                # Perform next action
                action = self.choose_action()
                self.states.append(self.Environment.next_position(action))
                self.Environment = self.take_action(action)
                self.Environment.episode_is_over()
                
                # Store new state and reward
                new_reward = self.Environment.give_reward()
                new_state = self.Environment.state

                # Update state values in policy
                self.policy = self.PolicyMaker.update_policy(
                    cur_state,
                    new_reward,
                    new_state,
                    policy=self.policy
                )
        return self.policy

    def test(self):
        old_exploration_rate = self.config.exploration_rate
        self.config.exploration_rate = 0.0

        while not self.Environment.episode_over:
            # Perform next action
            action = self.choose_action()
            self.states.append(self.Environment.next_position(action))
            self.Environment = self.take_action(action)
            self.Environment.episode_is_over()
            self.Environment.give_reward()
    
        self.games.append(self.states)
        self.config.exploration_rate = old_exploration_rate
        print('Last game: ', self.games[-1])
        print('Reward: ', self.Environment.current_reward)

class Environment:
    """ 
    This class defines the environment which the Agent learns. `obstacles`
    refers to states in the environment with reward equal to
    `ObstacleReward`, and `severe_obstacles` refers to states in the environment with
    reward equal to `SevereObstacle`. For further details consult the confif
    `grid_world_config.ini`.
      
    Attributes: 
        config (dict): State and reward parameters.
    """
    def __init__(self, config):
        """ 
        The constructor for the State class. 
  
        Parameters: 
            config (dict): State and reward parameters.  
        """
        self.config = config
        self.current_reward = 0.0
        self.rows = self.config.rows
        self.cols = self.config.cols
        self.action_reward = self.config.action_reward
        self.start_state = self.config.start_state
        self.win_state = self.config.win_state
        self.win_reward = self.config.win_reward
        self.obstacles = self.config.obstacles
        self.obstacles_reward = self.config.obstacles_reward
        self.severe_obstacles = self.config.severe_obstacles
        self.severe_obstacles_reward = self.config.severe_obstacles_reward

        # Define current board.
        self.board = np.zeros([self.rows, self.cols])
        self.state = self.start_state   
        self.episode_over = False

    def give_reward(self):
        """
        Based on the current state a reward is returned.

        Returns: 
            current_reward (int): A reward based on the state.
        """

        self.current_reward += self.action_reward
        if self.state in self.obstacles:
            self.current_reward += self.obstacles_reward
        if self.state in self.severe_obstacles:
            self.current_reward += self.severe_obstacles_reward
        return self.current_reward

    def next_position(self, action):
        """
        Based on chosen action returns next position.

        Parameters: 
            action (str): A string determining the next action.
          
        Returns: 
            state (tuple): A tuple of ints determining the next state.
        """
        if action == "up":
            next_state = (self.state[0] - 1, self.state[1])
        elif action == "down":
            next_state = (self.state[0] + 1, self.state[1])
        elif action == "left":
            next_state = (self.state[0], self.state[1] - 1)
        else:
            next_state = (self.state[0], self.state[1] + 1)

        # Checks if in environment
        if (next_state[0] >= 0) and (next_state[0] <= self.rows-1):
            if (next_state[1] >= 0) and (next_state[1] <= self.cols-1):
                return next_state
        # If not on board, return current state
        return self.state	               

    def episode_is_over(self):
        """
        Checks if the action cap is exceeded or if in a winning position.
        """
        win = (self.state == self.win_state) 
        if win:
            self.episode_over = True