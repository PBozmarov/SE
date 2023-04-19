from environment.dummy_env_rectangular_pin import DummyPlacementEnv
from environment.wrapper import (
    FlatteningActionWrapper,
    FlatteningActionMaskObservationWrapper,
)
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
import ray
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.algorithms.ppo.ppo import PPOConfig

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict, Tuple

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class Model(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, None, model_config, name)
        grid_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["height"],
                model_config["custom_model_config"]["width"],
            ),
            dtype=tf.float32,
            name="grid_input",
        )
        component_feature_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                model_config["custom_model_config"]["component_feature_vector_width"],
            ),
            dtype=tf.float32,
            name="component_feature_input",
        )
        component_pin_feature_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                model_config["custom_model_config"]["max_num_pins_per_component"],
                model_config["custom_model_config"]["pin_feature_vector_width"],
            ),
            dtype=tf.float32,
            name="pin_feature_input",
        )
        placement_mask_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                1
            ),
            dtype=tf.float32,
            name="placement_mask_input",
        )
        processed_grid = tfkl.Conv2D(filters=3, kernel_size=3)(
            tf.expand_dims(grid_input, axis=-1)
        )
        processed_grid = tfkl.BatchNormalization()(processed_grid)
        processed_grid = tf.nn.relu(processed_grid)
        processed_grid = tfkl.Conv2D(filters=3, kernel_size=3)(processed_grid)
        processed_grid = tfkl.BatchNormalization()(processed_grid)
        processed_grid = tf.nn.relu(processed_grid)
        #processed_grid = tfkl.Conv2D(filters=16, kernel_size=3)(processed_grid)
        #processed_grid = tfkl.BatchNormalization()(processed_grid)
        #processed_grid = tf.nn.relu(processed_grid)
        #processed_grid = tfkl.Conv2D(filters=32, kernel_size=3)(processed_grid)
        #processed_grid = tfkl.BatchNormalization()(processed_grid)
        #processed_grid = tf.nn.relu(processed_grid)

        # create component and pin encoding matrix
        component_encoding_dim = 16
        pin_encoding_dim = 16
        components_encoding = tfkl.Dense(component_encoding_dim)(component_feature_input)
        # encode all the pins for a component into a single vector
        # initialise encoding layer
        pin_encoding_layer = tfkl.Dense(pin_encoding_dim)
        component_pin_encodings = []
        #print(component_pin_feature_input.shape)
        for i in range(model_config["custom_model_config"]["max_num_components"]):
            pins_encoding = pin_encoding_layer(component_pin_feature_input[:, i])
            # sum up all the pins for a component
            pins_encoding = tf.reduce_sum(pins_encoding, axis=1)
            # append to list
            component_pin_encodings.append(pins_encoding)
        component_pin_encodings = tf.stack(component_pin_encodings, axis=0)
        component_pin_encodings = tf.transpose(component_pin_encodings, perm=[1, 0, 2])

        # concatenate the component and pin encodings with the placement mask
        processed_component_pin_encodings = tf.concat(
            [components_encoding, component_pin_encodings, placement_mask_input], axis=2
        )

        # create the attention layer
        hidden_size = 16
        query_comp = tfkl.Dense(hidden_size)(processed_component_pin_encodings)
        key_comp = tfkl.Dense(hidden_size)(processed_component_pin_encodings)
        value_comp = tfkl.Dense(hidden_size)(processed_component_pin_encodings)

        attention_weights_comp = tf.matmul(query_comp, key_comp, transpose_b=True)
        attention_weights_comp = tf.nn.softmax(attention_weights_comp, axis=-1)

        attention_output_comp = tf.matmul(attention_weights_comp, value_comp)
        #dense_outputs_comp = tfkl.Dense(1)(attention_output_comp)
        dense_outputs_comp = tf.nn.relu(attention_output_comp)

        encoding_grid = tfkl.Flatten()(processed_grid)
        encoding_component_pins = tfkl.Flatten()(dense_outputs_comp)
        encoding = tfkl.Concatenate()([encoding_grid, encoding_component_pins])
    
        logits = tfkl.Dense(action_space.n)(encoding)
        #logits = tf.nn.softmax(logits)
        value = tfkl.Dense(1)(encoding)
        self.model = tf.keras.Model(
            [grid_input, component_feature_input, component_pin_feature_input, placement_mask_input], [logits, value], name="action_model"
        )

    def forward(self, input_dict, state, seq_lens):
        placement_mask = tf.cast(input_dict["obs"]["placement_mask"], tf.float32)
        component_mask = tf.cast(input_dict["obs"]["component_mask"], tf.float32)
        all_components_feature = input_dict["obs"]["all_components_feature"]
        all_pins_num_feature = input_dict["obs"]["all_pins_num_feature"]
        all_pins_cat_feature = tf.cast(input_dict["obs"]["all_pins_cat_feature"], tf.int32)

        # current_component_id = tf.cast(input_dict["obs"]["current_component_id"], tf.int32)[0]
        max_num_nets = 3
        max_num_pins_per_comp = 4#tf.cast(all_pins_num_feature.shape[1] // all_components_feature.shape[1], tf.int32)
        max_num_components = 5#tf.cast(all_components_feature.shape[1], tf.int32)

        pins_cat_one_hot = tf.one_hot(all_pins_cat_feature[:, :, 0], 3 +1)
        # make last row of pins_cat_one_hot to be all zeros
        #pins_cat_one_hot_empty_pin = tf.zeros([1, pins_cat_one_hot.shape[-1]])

        #pins_cat_one_hot[:,-1,:] = tf.zeros([1, pins_cat_one_hot.shape[-1]])
        #pins_cat_one_hot = tf.concat([pins_cat_one_hot, pins_cat_one_hot_empty_pin], axis=1)
        pins_cat_one_hot = tf.ensure_shape(pins_cat_one_hot, [None, pins_cat_one_hot.shape[1], 3 +1])
        all_pins_feature = tf.concat([all_pins_num_feature, pins_cat_one_hot], axis=-1)
        # add another row to all_pins_feature to represent the empty pin
        empty_pin_feature = tf.zeros([1, 10])
        
        all_pins_feature = tf.concat([all_pins_num_feature, pins_cat_one_hot], axis=2)

        # extract pin_ids which are the columns from 5 to the end in all_components_feature
        pin_ids = all_components_feature[:, :, 5:]
        # convert pin_ids to int
        pin_ids = tf.cast(pin_ids, tf.int32)
        # change all -1 to max_num_pins_per_comp+1
        pin_ids = tf.where(tf.equal(pin_ids, -1), max_num_pins_per_comp*max_num_components, pin_ids)
        # for each componrnt, extract the pins from all_pins_feature
        all_component_pins_feature = tf.gather(all_pins_feature, pin_ids, batch_dims=1)

        logits, self._value_out = self.model(
            [input_dict["obs"]["grid"], all_components_feature[:, :, :5], all_component_pins_feature, placement_mask]
        )
        logits += tf.maximum(
            tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min
        )

        return logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def create_env(env_config):

    env = DummyPlacementEnv(
        env_config["height"],
        env_config["width"],
        env_config["net_distribution"],
        env_config["pin_spread"],
        env_config["min_component_w"],
        env_config["max_component_w"],
        env_config["min_component_h"],
        env_config["max_component_h"],
        env_config["max_num_components"],
        env_config["min_num_components"],
        env_config["min_num_nets"],
        env_config["max_num_nets"],
        env_config["max_num_pins_per_net"],
        env_config["min_num_pins_per_net"],
        env_config["reward_type"],
        env_config["reward_beam_width"],
        env_config["weight_wirelength"],
    )
    env = FlatteningActionMaskObservationWrapper(env)
    env = FlatteningActionWrapper(env)
    return env


class CustomCallbackClass(DefaultCallbacks):
    """Example of a custom callback that logs wirelength and the
       number of intersections at the end of each episode."""
    def on_episode_end(self, 
                        worker, 
                        base_env, 
                        policies, 
                        episode, 
                        **kwargs):
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forw
        """
        # Get last info dict from the episode
        info = episode.last_info_for('agent0')
        wirelength = info["wirelength"]
        num_ints = info["num_intersections"]

        # Store the wirelength and num_ints in the episode's custom metrics
        episode.custom_metrics["normalized_wirelengths"] = wirelength
        episode.custom_metrics["num_intersections"] = num_ints


if __name__ == "__main__":
    tune.register_env("dummy_placement_env", create_env)
    ModelCatalog.register_custom_model("my_model", Model)
    PPO_config = PPOConfig().callbacks(CustomCallbackClass)
    config = PPO_config
    #config.training(lr=0.0001, train_batch_size=256, sgd_minibatch_size=256, optimizer={"type": "adam", "epsilon": 1e-5})
    config["model"]["custom_model"] = "my_model"
    config["model"]["custom_model_config"]["height"] = 10
    config["model"]["custom_model_config"]["width"] = 10
    config["model"]["custom_model_config"]["max_num_components"] = 5
    config["model"]["custom_model_config"]["min_num_components"] = 5
    config["model"]["custom_model_config"]["max_component_w"] = 2
    config["model"]["custom_model_config"]["max_component_h"] = 2
    config["model"]["custom_model_config"]["max_num_pins_per_component"] = 2*2
    config["model"]["custom_model_config"]["component_feature_vector_width"] = 5
    config["model"]["custom_model_config"]["pin_feature_vector_width"] = 4 + 4

    config["env"] = "dummy_placement_env"
    config["env_config"]["height"] = 10
    config["env_config"]["width"] = 10
    config["env_config"]["net_distribution"] = 9
    config["env_config"]["pin_spread"] = 9
    config["env_config"]["min_component_w"] = 2
    config["env_config"]["max_component_w"] = 2
    config["env_config"]["min_component_h"] = 2
    config["env_config"]["max_component_h"] = 2
    config["env_config"]["max_num_components"] = 5
    config["env_config"]["min_num_components"] = 5
    config["env_config"]["min_num_nets"] = 3
    config["env_config"]["max_num_nets"] = 3
    config["env_config"]["max_num_pins_per_net"] = 6
    config["env_config"]["min_num_pins_per_net"] = 6
    config["env_config"]["reward_type"] = "centroid"
    config["env_config"]["reward_beam_width"] = 2
    config["env_config"]["weight_wirelength"] = 1.0

    ray.init(local_mode=True)
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 100},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        keep_checkpoints_num=5,
        restore=None,
    )
