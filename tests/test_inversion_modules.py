import numpy as np

from py12box_invert.inversion_modules import difference_operator


def test_difference_operator():

    for freq in [1, 4, 12]:

        nyears=10

        # Some fake emissions.
        # Increasing values in each box.
        # Difference rate of increase in each box
        emissions = np.vstack([np.arange(0., 1000., 100./freq),
                                np.arange(0., 100., 10./freq),
                                np.arange(0., 10., 1./freq),
                                np.arange(0., 1., 0.1/freq)]).T

        x = emissions.flatten()

        # Calculate emissions difference using difference operator
        difference = difference_operator(len(x), freq) @ x

        # Reshape to time x box array
        emissions_difference = difference.reshape((int(len(x)/4), 4))

        # Manually calculate the difference in the emissions array
        actual_diffs = np.vstack([emissions[ti + freq, :] - emissions[ti, :] for ti in range(freq*(nyears-1))])

        # Check that all differences are correct up to last year
        assert np.allclose(actual_diffs, emissions_difference[:freq*(nyears-1), :])
        # Final year should only contain zeros
        assert np.sum(emissions_difference[(nyears-1)*freq:, :]) == 0.