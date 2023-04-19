import gym  # type: ignore
import numpy as np


class FlatteningActionMaskObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, gym.spaces.Dict)
        # for v in env.observation_space.spaces.values():
        #     assert isinstance(v, gym.spaces.Box)
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.utils.flatten_space(v) if k == "action_mask" else v
                for k, v in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation):
        return {
            k: gym.spaces.utils.flatten(self.observation_space[k], v)
            if k == "action_mask"
            else v
            for k, v in observation.items()
        }


class FlatteningActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        self.factor_sizes = [space.n for space in env.action_space.spaces]
        self.action_space = gym.spaces.Discrete(np.prod(self.factor_sizes))

    def action(self, action):
        """Translate discrete action into tuple of discrete actions"""
        unflattened_action = []
        self.factor_sizes[0] = self.factor_sizes[1] * self.factor_sizes[2]
        action, remainder = divmod(action, self.factor_sizes[0])
        unflattened_action.append(action)

        action_x, action_y = divmod(remainder, self.factor_sizes[2])
        unflattened_action.extend([action_x, action_y])

        return unflattened_action

    def validate_action(self, action):
        return self.env.validate_action(*self.action(action))
