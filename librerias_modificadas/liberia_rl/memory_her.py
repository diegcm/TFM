from collections import deque, namedtuple
import warnings
import random
import copy
from itertools import chain #Cambiado por mi. Antes: nada.

import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1, goal, flag, ang_vel_array, info') 


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high

        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick

        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.

        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation

    # Argument
        observation (list): List of observation
    
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    if isinstance(observation, dict):
        keys = observation.keys()
        obs = dict()
        for key in keys:
            obs[key] = np.zeros(observation[key].shape)
        return obs
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory:
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_states = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, state, action, reward, next_state, terminal, goal, flag, ang_vel_array, info, training=True):
        self.recent_states.append(state)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_state):
        """Return list of last observations

        # Argument
            current_observation (object): Last observation

        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        #print('Current_state: ', current_state)
        state = current_state #Cambiado por mí. Original: [current_observation]
        idx = len(self.recent_states) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_states[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, initial_dict, **kwargs):
        super().__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.states = deque(maxlen=limit)
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.next_states = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.goals = deque(maxlen=limit)
        self.flags = deque(maxlen=limit)
        self.ang_vel_arrays = deque(maxlen=limit)
        self.infos = deque(maxlen=limit)

        if initial_dict is not None: #Se inicializa la memoria con experiencias guardadas en Callback
            self.states = initial_dict['self.states']
            self.actions = initial_dict['self.actions']
            self.rewards = initial_dict['self.rewards']
            self.next_states = initial_dict['self.next_states']
            self.terminals = initial_dict['self.terminals']
            self.goals = initial_dict['self.goals']
            self.flags = initial_dict['self.flags']
            self.ang_vel_arrays = initial_dict['self.ang_vel_arrays']
            self.infos = initial_dict['self.infos']


    def sample_goals(self):

        goals = []
        #nb_goals = min(len(self.rewards, 250))
        nb_goals = 250

        rewards_list = np.array(list(self.rewards)) #Ignorar las recompensas asociadas a estados terminales con goal=None
        for index in range(0, len(rewards_list)-1):
            if self.goals[index] is None:
                rewards_list[index] = -np.inf #Estos valores no entrarán en la lista de mayores recompensas

        # Obtener los índices de las recompensas más altas
        highest_rewards_indexes = np.argsort(rewards_list)[-2:].tolist() #-int(nb_goals/2):

        for index in highest_rewards_indexes: #Añade goles asociados a recompensas máximas
            #if self.goals[index] is not None:
            goals.append(self.goals[index])

        while len(goals) < nb_goals: #Completa la lista con goles aleatorios no correspondientes a estados terminales
            index = random.randint(0, len(self.goals)-1)
            if index not in highest_rewards_indexes and self.goals[index] is not None:
                goals.append(self.goals[index])

        return goals


    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = self.states[idx - 1] #Cambiado por mi. Antes: [self.observations[idx - 1]]
            
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.states[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            goal = self.goals[idx - 1]
            flag = self.flags[idx - 1]
            ang_vel_array = self.ang_vel_arrays[idx - 1]
            info = self.infos[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.

            #state1 = [copy.deepcopy(x) for x in state0[1:]]
            #state1.append(self.observations[idx].tolist()) #Cambiado por mi. Antes: state1.append(self.observations[idx])

            state1 = self.next_states[idx - 1]

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)

            #print('state0:', state0) #Cambiado por mí. Antes: nada.
            #state0 = list(chain.from_iterable(state0)) #Cambiado por mi. Antes: nada
            state0=state0[0] #Cambiado por mí. Antes: nada.

            #print('state1:', state1) #Cambiado por mí. Antes: nada.
            #for _ in range(1,3): #Cambiado por mi. Antes: nada
                #state1 = list(chain.from_iterable(state1)) #Cambiado por mi. Antes: nada

            state1=state1[0] #Cambiado por mí. Antes: nada.


            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1, goal=goal, 
                                          flag=flag, ang_vel_array = ang_vel_array, info=info))
                                          
        assert len(experiences) == batch_size
        
        return experiences

    def append(self, state, action, reward, next_state, terminal, goal, flag, ang_vel_array, info, training=True):  #state , action, reward, next_state (obs), done, goal
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super().append(state, action, reward, next_state, terminal, goal, flag, ang_vel_array, info, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.terminals.append(terminal)
            self.goals.append(goal)
            self.flags.append(flag)
            self.ang_vel_arrays.append(ang_vel_array)
            self.infos.append(info)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.states)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super().get_config()
        config['limit'] = self.limit
        return config

    def return_deque_dict(self):
         
        deque_dict = {'self.states': self.states, 'self.actions': self.actions,
        'self.rewards': self.rewards, 'self.next_states': self.next_states,
        'self.terminals': self.terminals, 'self.goals': self.goals, 'self.flags': self.flags, 
        'self.ang_vel_arrays': self.ang_vel_arrays, 'self.infos': self.infos}

        return deque_dict

class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

        self.params = deque(maxlen=limit)
        self.intermediate_rewards = []
        self.total_rewards = deque(maxlen=limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of params and rewards

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of params randomly selected and a list of associated rewards
        """
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        """Append a reward to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super().append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        """Closes the current episode, sums up rewards and stores the parameters

        # Argument
            params (object): Parameters associated with the episode to be stored and then retrieved back in sample()
        """
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        """Return number of episode rewards

        # Returns
            Number of episode rewards
        """
        return len(self.total_rewards)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    

