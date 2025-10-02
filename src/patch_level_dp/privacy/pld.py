import math
import numbers
from typing import Iterable, List, Sequence, Union

import numpy as np
import scipy
from scipy import stats
from scipy.special import logsumexp

from dp_accounting.pld import common
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)
from dp_accounting.pld.privacy_loss_mechanism import (
    AdditiveNoisePrivacyLoss, AdjacencyType, ConnectDotsBounds,
    MixtureGaussianPrivacyLoss, TailPrivacyLossDistribution)
from dp_accounting.pld.pld_pmf import SparsePLDPmf


class SwitchingPrivacyLoss(AdditiveNoisePrivacyLoss):
    # TODO: Do fancy convex conjugate thing from Zhu et al. 2022
    # for all the NotImplementedError methods.

    def __init__(self,
                 epsilon_threshold: float,
                 below_threshold_pl: AdditiveNoisePrivacyLoss,
                 above_threshold_pl: AdditiveNoisePrivacyLoss):

        self.epsilon_threshold = epsilon_threshold
        self.below_threshold_pl = below_threshold_pl
        self.above_threshold_pl = above_threshold_pl

        if below_threshold_pl.discrete_noise != above_threshold_pl.discrete_noise:
            raise ValueError('PLs must be both discrete or both continuous.')

        self.discrete_noise = below_threshold_pl.discrete_noise

        if self.discrete_noise:
            raise NotImplementedError('Only continuous PLs supported currently.')

        if not np.isclose(
                below_threshold_pl.get_delta_for_epsilon(epsilon_threshold),
                above_threshold_pl.get_delta_for_epsilon(epsilon_threshold)):
            raise ValueError('Tradeoff functions must intersect at epsilon_threshold.')

    def mu_upper_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def mu_lower_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def get_delta_for_epsilon(
            self, epsilon: Union[float, List[float]]) -> Union[float, List[float]]:

        is_scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
        deltas = np.zeros_like(epsilons, dtype=float)

        below_threshold_mask = (epsilons < self.epsilon_threshold)
        above_threshold_mask = ~below_threshold_mask

        if below_threshold_mask.sum() > 0:
            deltas[below_threshold_mask] = self.below_threshold_pl.get_delta_for_epsilon(
                epsilons[below_threshold_mask])

        if above_threshold_mask.sum() > 0:
            deltas[above_threshold_mask] = self.above_threshold_pl.get_delta_for_epsilon(
                epsilons[above_threshold_mask])

        return float(deltas[0]) if is_scalar else deltas

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        below_threshold_bounds = self.below_threshold_pl.connect_dots_bounds()
        above_threshold_bounds = self.above_threshold_pl.connect_dots_bounds()

        epsilon_upper = max(below_threshold_bounds.epsilon_upper,
                            above_threshold_bounds.epsilon_upper)

        epsilon_lower = min(below_threshold_bounds.epsilon_lower,
                            above_threshold_bounds.epsilon_lower)

        return ConnectDotsBounds(epsilon_upper=epsilon_upper,
                                 epsilon_lower=epsilon_lower)

    def privacy_loss(self, x: float) -> float:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss_without_subsampling(self, privacy_loss: float) -> float:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> 'AdditiveNoisePrivacyLoss':
        raise NotImplementedError('SwitchingPL is currently only meant for use with connect_the_dots.')


class PldPmf(SparsePLDPmf):
  """Modified version of SparsePLDPmf with altered self_compose method."""

  def self_compose(self,
                   num_times: int,
                   tail_mass_truncation: float = 1e-15) -> 'PLDPmf':
    """See base class."""
    if num_times <= 0:
      raise ValueError(f'num_times should be >= 1, num_times={num_times}')
    if num_times == 1:
      return self

    # Compute a rough upper bound overestimate, since from some power, the PMF
    # becomes dense and start growing linearly further. But in this case we
    # should definitely go to dense.

    return self.to_dense_pmf().self_compose(num_times, tail_mass_truncation)
    