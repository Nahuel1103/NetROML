import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from utils import graphs_to_tensor, nuevo_get_rates

class WifiEnv(gym.Env):
    """Gymnasium environment integrado con tus funciones.

    Action encoding (flattened vector length = num_links * 3): for each link i:
      [ tx_on (0/1), channel_idx (0..num_channels-1), power_idx (0..power_levels-1) ]

    Observation: channel_matrix (num_links x num_links) -- dtype float32
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 num_links=5,
                 num_channels=5,
                 power_levels=3,
                 max_antenna_power_dbm=6,
                 sigma=1e-3,
                 graphs=None,
                 graphs_base_path=None,
                 train=True,
                 max_steps=1):
        super(WifiEnv, self).__init__()

        self.num_links = num_links
        self.num_channels = num_channels
        self.power_levels = power_levels  # integer count
        self.max_antenna_power_dbm = max_antenna_power_dbm
        self.max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
        self.sigma = sigma
        self.max_steps = max_steps

        # power levels as multiples: [pmax/P, 2pmax/P, ..., pmax]
        step = self.max_antenna_power_mw / float(self.power_levels)
        self.power_levels_mw = (torch.arange(1, self.power_levels + 1, dtype=torch.float) * step)

        # Action space: vectorized per-link [tx, channel, power]
        self.action_space = spaces.MultiDiscrete([(2, self.num_channels, self.power_levels)]*self.num_links)

        # Observation: channel matrix
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_links, self.num_links), dtype=np.float32)

        # Dataset
        self.channel_matrix_tensor = None
        if graphs is not None:
            # graphs expected to be (x_tensor, channel_matrix_tensor) as returned by graphs_to_tensor
            self.x_tensor, self.channel_matrix_tensor = graphs
        else:
            # try to load from path if provided
            try:
                self.x_tensor, self.channel_matrix_tensor = graphs_to_tensor(train=train, num_links=self.num_links, base_path=graphs_base_path)
            except Exception:
                self.x_tensor, self.channel_matrix_tensor = None, None

        self.idx = 0
        self.step_count = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if self.channel_matrix_tensor is not None:
            # take current idx
            self.state = self.channel_matrix_tensor[self.idx % len(self.channel_matrix_tensor)].float()
            # keep idx as-is; step() will advance
        else:
            self.state = torch.rand((self.num_links, self.num_links), dtype=torch.float)

        return self.state.numpy().astype(np.float32), {}

    def step(self, action):
        # action: flatten vector produced by MultiDiscrete -> reshape to [num_links, 3]
        action = np.asarray(action, dtype=np.int64).reshape(self.num_links, 3)

        phi = torch.zeros((self.num_links, self.num_channels), dtype=torch.float)
        for i in range(self.num_links):
            tx_on, ch, p_idx = int(action[i, 0]), int(action[i, 1]), int(action[i, 2])
            if tx_on == 1:
                # safety clamps (shouldn't be necessary if using action_space)
                if ch < 0 or ch >= self.num_channels:
                    ch = ch % self.num_channels
                if p_idx < 0 or p_idx >= self.power_levels:
                    p_idx = max(0, min(p_idx, self.power_levels - 1))
                phi[i, ch] = self.power_levels_mw[p_idx]

        # batch dims
        phi_b = phi.unsqueeze(0)  # [1, links, channels]
        channel_b = self.state.unsqueeze(0)  # [1, links, links]

        rates = nuevo_get_rates(phi_b, channel_b, self.sigma)  # [1, links]
        reward = float(torch.sum(rates).item())

        # advance dataset index and update state if we have dataset
        if self.channel_matrix_tensor is not None:
            self.idx = (self.idx + 1) % len(self.channel_matrix_tensor)
            self.state = self.channel_matrix_tensor[self.idx].float()

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {"rates": rates.detach().cpu().numpy(), "phi": phi.detach().cpu().numpy()}

        return self.state.numpy().astype(np.float32), reward, done, False, info

    def render(self):
        print("Current channel matrix (state):\n", self.state)

    def close(self):
        return
