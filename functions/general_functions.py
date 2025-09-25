# imports
import numpy as np
import rft1d
import scipy
from scipy.stats import norm

from seismostats.utils import simulate_magnitudes_binned, bin_to_precision
from seismostats.analysis import (
    b_value_to_beta,
    estimate_b,
)


def dist_to_ref(x,  x_ref, y, y_ref, z=None, z_ref=None):
    if z is None:
        return np.sqrt((x - x_ref)**2 + (y - y_ref)**2)
    else:
        return np.sqrt((x - x_ref)**2 + (y - y_ref)**2 + (z - z_ref)**2)


def likelihood_exp(
        magnitude: np.ndarray,
        mc: float | np.ndarray,
        b_value:  float | np.ndarray) -> np.ndarray:
    """likelihood of each magnitude given the b-value and the completeness

    Args:
        magnitude:  array of magnitudes
        mc:         completeness magnitude. if a single value is given, it is
                assumed that the completeness magnitude is the same for all
                magnitudes.
        b_value:    b-value. if a single value is given, it is assumed that
                the b-value is the same for all magnitudes.
    """
    beta = b_value_to_beta(b_value)
    p = beta * np.exp(-beta * (magnitude - mc))
    return p


def update_welford(existing_aggregate: tuple, new_value: float) -> tuple:
    """Update Welford's algorithm for computing a running mean and standard
    deviation. Suited for both scalar and array inputs. nan values are
    ignored.

    Args:
        existing_aggregate:     (count, mean, M2) where count is the number
                        of values used up tp that point, mean is the mean and
                        M2 is the sum of the squares of the differences from
                        the mean of the previous step
        new_value:              new value of the series of which the standard
                        deviation and mean is to be calculated

    Returns:
        (updated_count, updated_mean, updated_M2)
    """
    count, mean, M2 = existing_aggregate

    # Convert to np array if not already
    new_value = np.asarray(new_value)
    mean = np.asarray(mean)
    M2 = np.asarray(M2)

    if np.isscalar(count):
        count = np.array(count)

    # Identify valid components
    valid = ~np.isnan(new_value)

    # Increment count only where valid
    count_new = np.where(valid, count + 1, count)

    # Compute deltas only for valid components
    delta = np.zeros_like(mean)
    delta2 = np.zeros_like(mean)

    delta[valid] = new_value[valid] - mean[valid]
    mean_new = mean.copy()
    mean_new[valid] += delta[valid] / count_new[valid]
    delta2[valid] = new_value[valid] - mean_new[valid]
    M2_new = M2.copy()
    M2_new[valid] += delta[valid] * delta2[valid]

    return (count_new, mean_new, M2_new)


def finalize_welford(existing_aggregate: tuple) -> tuple[float, float]:
    """Retrieve the mean, variance and sample variance from an aggregate.

    Args:
        existing_aggregate:  (count, mean, M2) where count is the number
                        of values used up tp that point, mean is the mean and
                        M2 is the sum of the squares of the differences for
                        the whole series of which the standard deviation and
                        mean is to be calculated

    Returns:
        (mean, variance)
    """
    count, mean, M2 = existing_aggregate

    mean = np.asarray(mean)
    M2 = np.asarray(M2)
    count = np.asarray(count)

    variance = np.full_like(mean, np.nan)

    valid = count > 1
    # use count - 1 for sample variance if needed
    variance[valid] = M2[valid] / count[valid]

    return mean, variance


def transform_n(
    x: np.ndarray, b: float, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """transform b-value to be comparable to other b-values

    Args:
        x (float):  b-value to be transformed
        b (float):  true b-value
        n1 (int):   number of events in the distribution to be transformed
        n2 (int):   number of events to which the distribution is transformed

    Returns:
        x (float):  transformed b-value
    """
    x_transformed = b / (1 - np.sqrt(n1 / n2) * (1 - b / x))
    return x_transformed


def inverse_norm(x: np.ndarray, b: float, n: int) -> np.ndarray:
    """distribution function of the reciprocal gaussian distribution. This is
    the distribution of 1/X where X is normally distributed. It is designed
    specifically to be used as proxy of the distribution of b-value estiamtes.

    Args:
        x:      values for which the distribution function is calculated
            (i.e. estimated b-value)
        b:      true b-value
        n:      number of events in the distribution

    Returns:
        dist:   probability density at x
    """
    dist = (
        1
        / b
        / np.sqrt(2 * np.pi)
        * np.sqrt(n)
        * (b / x) ** 2
        * np.exp(-n / 2 * (1 - b / x) ** 2)
    )
    return dist


class inverse_norm_class(scipy.stats.rv_continuous):
    """distribution function of the reciprocal normal distribution.This can be
    used, for instance to
    - compute the cdf
    - generate random numbers that follow the reciprocal normal distribution

    Args:
        b:      true b-value
        n_b:    number of events in the distribution
    """

    def __init__(self, b, n_b):
        scipy.stats.rv_continuous.__init__(self, a=0.0)
        self.b_val = b
        self.n_b = n_b

    def _pdf(self, x):
        return inverse_norm(x, b=self.b_val, n=self.n_b)


def cdf_inverse_norm(x: np.ndarray, b: float, n_b: int) -> np.ndarray:
    """distribution function of the reciprocal gaussian distribution. This is
    the distribution of 1/X where X is normally distributed. It is designed
    specifically to be used as proxy of the distribution of b-value estiamtes.

    Args:
        x:      values for which the distribution function is calculated
            (i.e. estimated b-value)
        b:      true b-value
        n:      number of events in the distribution

    Returns:
        y:   cdf at x
    """

    x = np.sort(x)
    x = np.unique(x)
    y = np.zeros(len(x))
    inverse_normal_distribution = inverse_norm_class(b=b, n_b=n_b)
    y = inverse_normal_distribution.cdf(x=x)

    return x, y


def simulate_rectangular(
    n_total: int,
    n_deviation: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with a step of length N_deviation in the
    b-value

    Args:
        n_total:        total number of magnitudes to simulate
        n_deviation:    number of magnitudes with deviating b-value
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    n_loop1 = int((n_total - n_deviation) / 2)

    b_true = np.ones(n_total) * b
    b_true[n_loop1: n_loop1 + n_deviation] = b + delta_b  # noqa

    magnitudes = simulate_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_step(
    n_total: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
    idx_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with a step at idx in the b-value

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        idx_step:       index of the magnitude where the step occurs. if None,
                    the step occurs at the middle of the sequence

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """

    if idx_step is None:
        idx_step = int(n_total / 2)

    b_true = np.ones(n_total) * b
    b_true[idx_step:] = b + delta_b

    magnitudes = simulate_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_sinus(
    n_total: int,
    n_wavelength: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with an underlying sinusoidal b-value
    distribution

    Args:
        n_total:        total number of magnitudes to simulate
        n_wavelength:   wavelength of the sinusoidal
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    b_true = (
        b
        + np.sin(np.arange(n_total) / (n_wavelength - 1) * 2 * np.pi) * delta_b
    )

    magnitudes = simulate_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_ramp(
    n_total: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with an underlying b-value that rises
    constantly

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    b_true = b + np.arange(n_total) / n_total * delta_b

    magnitudes = simulate_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_randomfield(
    n_total: int,
    kernel_width: float,
    b: float,
    b_std: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes where the underlying b-values vary with time
    as a random gaussian process

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        idx_step:       index of the magnitude where the step occurs. if None,
                    the step occurs at the middle of the sequence

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    magnitudes = np.zeros(n_total)
    kernel_width
    b_s = abs(b + rft1d.random.randn1d(1, n_total, kernel_width) * b_std)

    for ii in range(n_total):
        magnitudes[ii] = simulate_magnitudes_binned(
            1, b_s[ii], mc, delta_m,
        ).item()
    return bin_to_precision(magnitudes, delta_m), b_s


def utsu_test(
    b1: np.ndarray, b2: np.ndarray, n1: np.ndarray[int], n2: np.ndarray
) -> np.ndarray:
    """Given two b-value estimates from two magnitude samples, this functions
    gives back the probability that the actual underlying b-values are not
    different. All the input arrays have to have the same length.

    Source: TODO Need to verify that this is used in Utsu 1992 !!!

    Args:
        b1:     b-value estimate of first sample
        b2:     b-value estimate of seconds sample
        N1:     number of magnitudes in first sample
        N2:     number of magnitudes in second sample

    Returns:
        p:      Probability that the underlying b-value of the two samples is
            identical
    """
    delta_AIC = (
        -2 * (n1 + n2) * np.log(n1 + n2)
        + 2 * n1 * np.log(n1 + n2 * b1 / b2)
        + 2 * n2 * np.log(n2 + n1 * b2 / b1)
        - 2
    )
    p = np.exp(-delta_AIC / 2 - 2)
    return p


def normalcdf_incompleteness(
    mags: np.ndarray, mc: float, sigma: float
) -> np.ndarray:
    """Filtering function: normal cdf with a standard deviation of sigma. The
    output can be interpreted as the probability to detect an earthquake. At
    mc, the probability of detect an earthquake is per definition 50%.

    Args:
        mags:   array of magnitudes
        mc:     completeness magnitude
        sigma:  standard deviation of the normal cdf

    Returns:
        p:      array of probabilities to detect given earthquakes
    """
    p = np.array(len(mags))
    x = (mags - mc) / sigma
    p = norm.cdf(x)
    return p


def distort_completeness(
    mags: np.ndarray, mc: float, sigma: float
) -> np.ndarray:
    """
    Filter a given catalog of magnitudes with a given completeness magnitude
    with a filtering function that is a normal cdf with a standard deviation
    of sigma.

    Args:
        mags:   array of magnitudes
        mc:     completeness magnitude
        sigma:  standard deviation of the normal cdf

    Returns:
        mags:   array of magnitudes that passed the filtering function
    """
    p = normalcdf_incompleteness(mags, mc, sigma)
    p_test = np.random.rand(len(p))
    return mags[p > p_test]


def probability_m(
        a_value: float | np.ndarray,
        b_value: float | np.ndarray,
        m: float,
        m_ref: float = 0) -> float:
    """estimate the probability of an event larger than m

    Args:
        a_value:    a-value, scaled to the time of interest
        b_value:    b-value
        m:          magnitude at which the probability is estimated
        m_ref:      reference magnitude (at which the a-value is given), by
                default 0

    Returns:
        p:          probability of an event larger than m
    """
    n = 10 ** (a_value - b_value * (m - m_ref))
    p = 1 - np.exp(-n)
    return p


def b_synth(
    n: int,
    b: float,
    n_b: int,
    mc: float = 0,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> float:
    """create estaimted b-values from a given true b-value

    Args:
        n:              number of estimated beta / b-values to simulate
        b:              true beta / b-value
        n_b:            number of events per beta / b-value estimate
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        b_parameter:    'b_value' or 'beta'

    Returns:
        b_synth:    synthetic beta / b-value
    """

    mags = simulate_magnitudes_binned(
        n * n_b, b, mc, delta_m, b_parameter=b_parameter
    )

    b = np.zeros(n)
    for ii in range(n):
        b[ii] = estimate_b(
            mags[ii * n_b: (ii + 1) * n_b],  # noqa
            mc,
            delta_m,
            b_parameter=b_parameter,
        )
    return b


def empirical_cdf(
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cumulative distribution function (CDF)
    from a sample.

    Parameters:
        sample:     Magnitude sample
        mc:         Completeness magnitude, if None, the minimum of the sample
                is used
        delta_m:    Magnitude bin size, by default 1e-16. Its recommended to
                use the value that the samples are rounded to.
        weights:    Sample weights, by default None

    Returns:
        x:          x-values of the empirical CDF (i.e. the unique vector of
                magnitudes from mc to the maximum magnitude in the sample,
                binned by delta_m)
        y:          y-values of the empirical CDF (i.e., the empirical
                frequency observed in the sample corresponding to the x-values)
    """

    idx1 = np.argsort(sample)
    x = sample[idx1]
    x, y_count = np.unique(x, return_counts=True)
    y = np.cumsum(y_count) / len(sample)

    return x, y
