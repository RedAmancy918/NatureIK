from typing import Dict
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class NullLowdimRunner(BaseLowdimRunner):
    """
    A no-op runner that does not require any environment.
    Used to satisfy TrainDiffusionUnetLowdimWorkspace's requirement of env_runner.
    """
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir=output_dir)

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return {}