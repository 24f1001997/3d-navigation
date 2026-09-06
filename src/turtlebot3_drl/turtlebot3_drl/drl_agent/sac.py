#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .off_policy_agent import OffPolicyAgent, Network


class SACActor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(SACActor, self).__init__(name)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        x1 = torch.relu(self.fc1(states))
        x2 = torch.relu(self.fc2(x1))

        mean = self.mean(x2)

        log_std = self.log_std(x2)
        log_std = torch.clamp(log_std, -20, 2)

        if visualize and self.visual:
            self.visual.update_layers(
                states,
                mean,
                [x1, x2],
                [self.fc1.bias, self.fc2.bias]
            )

        return mean, log_std

    def sample(self, states):
        mean, log_std = self.forward(states)

        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        z = normal.rsample()

        # Squash action to [-1, 1]
        action = torch.tanh(z)

        # Correct log probability because of tanh transformation
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic(self, states):
        mean, _ = self.forward(states)
        return torch.tanh(mean)


class SACCritic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(SACCritic, self).__init__(name)

        # Q1
        self.q1_state = nn.Linear(state_size, hidden_size // 2)
        self.q1_action = nn.Linear(action_size, hidden_size // 2)
        self.q1_fc = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)

        # Q2
        self.q2_state = nn.Linear(state_size, hidden_size // 2)
        self.q2_action = nn.Linear(action_size, hidden_size // 2)
        self.q2_fc = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)

    def forward(self, states, actions):

        # Q1
        state_1 = torch.relu(self.q1_state(states))
        action_1 = torch.relu(self.q1_action(actions))

        x1 = torch.cat((state_1, action_1), dim=1)
        x1 = torch.relu(self.q1_fc(x1))
        q1 = self.q1_out(x1)

        # Q2
        state_2 = torch.relu(self.q2_state(states))
        action_2 = torch.relu(self.q2_action(actions))

        x2 = torch.cat((state_2, action_2), dim=1)
        x2 = torch.relu(self.q2_fc(x2))
        q2 = self.q2_out(x2)

        return q1, q2

    def Q1_forward(self, states, actions):
        state = torch.relu(self.q1_state(states))
        action = torch.relu(self.q1_action(actions))

        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.q1_fc(x))

        return self.q1_out(x)


class SAC(OffPolicyAgent):

    def __init__(self, device, sim_speed):
        super(SAC, self).__init__(device, sim_speed)

        # SAC parameters
        self.alpha = 0.2
        self.target_entropy = -float(self.action_size)

        # Automatic entropy tuning
        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.learning_rate
        )

        # Actor
        self.actor = self.create_network(
            SACActor,
            'actor'
        )

        self.actor_optimizer = self.create_optimizer(
            self.actor
        )

        # Twin critics
        self.critic = self.create_network(
            SACCritic,
            'critic'
        )

        self.critic_target = self.create_network(
            SACCritic,
            'target_critic'
        )

        self.critic_optimizer = self.create_optimizer(
            self.critic
        )

        # Initialize target critic
        self.hard_update(
            self.critic_target,
            self.critic
        )

        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_alpha_loss = 0.0

    def get_action(
        self,
        state,
        is_training,
        step,
        visualize=False
    ):

        state = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).to(self.device)

        if is_training:
            action, _ = self.actor.sample(
                state.unsqueeze(0)
            )

            action = action.squeeze(0)

        else:
            action = self.actor.deterministic(
                state.unsqueeze(0)
            ).squeeze(0)

        if visualize:
            self.actor(
                state.unsqueeze(0),
                visualize=True
            )

        return action.detach().cpu().numpy().tolist()

    def get_action_random(self):
        return np.random.uniform(
            -1.0,
            1.0,
            self.action_size
        ).tolist()

    def train(
        self,
        state,
        action,
        reward,
        state_next,
        done
    ):

        # ---------------------------------------------------------
        # 1. Target Q value
        # ---------------------------------------------------------

        with torch.no_grad():

            next_action, next_log_prob = self.actor.sample(
                state_next
            )

            target_q1, target_q2 = self.critic_target(
                state_next,
                next_action
            )

            target_q = torch.min(
                target_q1,
                target_q2
            )

            alpha = self.log_alpha.exp()

            target_value = (
                target_q -
                alpha * next_log_prob
            )

            q_target = (
                reward +
                (1.0 - done) *
                self.discount_factor *
                target_value
            )

        # ---------------------------------------------------------
        # 2. Critic update
        # ---------------------------------------------------------

        current_q1, current_q2 = self.critic(
            state,
            action
        )

        critic_loss = (
            F.smooth_l1_loss(current_q1, q_target) +
            F.smooth_l1_loss(current_q2, q_target)
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=2.0
        )

        self.critic_optimizer.step()

        # ---------------------------------------------------------
        # 3. Actor update
        # ---------------------------------------------------------

        new_action, log_prob = self.actor.sample(
            state
        )

        q1_new, q2_new = self.critic(
            state,
            new_action
        )

        q_new = torch.min(
            q1_new,
            q2_new
        )

        alpha = self.log_alpha.exp()

        actor_loss = (
            alpha * log_prob -
            q_new
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=2.0
        )

        self.actor_optimizer.step()

        # ---------------------------------------------------------
        # 4. Temperature / alpha update
        # ---------------------------------------------------------

        alpha_loss = -(
            self.log_alpha *
            (
                log_prob.detach() +
                self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # ---------------------------------------------------------
        # 5. Target critic update
        # ---------------------------------------------------------

        self.soft_update(
            self.critic_target,
            self.critic,
            self.tau
        )

        # Save losses
        self.last_actor_loss = actor_loss.detach().cpu()
        self.last_critic_loss = critic_loss.detach().cpu()
        self.last_alpha_loss = alpha_loss.detach().cpu()

        return [
            self.last_critic_loss,
            self.last_actor_loss
        ]