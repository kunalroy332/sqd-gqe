# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def n2_cfg():
    """The default Hydra config with the small N2 molecule, the cheapest real system."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs_sqd")):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(str(ROOT / "molecules" / "small" / "n2.yaml"))
    return cfg
