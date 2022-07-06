from bisect import bisect_right
import numpy as np
from scipy.optimize import minimize
import pymc as pm
import aesara.tensor as at
from patsy import dmatrix

from py12box_invert.utils import Store_model


def posterior_emissions(emissions_prior, freq_months, x_hat, P_hat,
                    calculate_uncertainty=True, from_zero=False):
    """Calculate linear adjustment to emissions

    Parameters
    ----------
    emissions_prior : ndarray
        A priori emissions estimate
    freq_moths : int
        Number of months in each emissions aggregation period (e.g. monthly, seasonal, annual)
    x_hat : ndarray
        Posterior emissions adjustment (Gg/yr)
    P_hat : ndarray
        Posterior emissions uncertainty covariance (Gg/yr)

    Returns 
    -------
    ndarray : 2d, n_months/freq_months x n_box
        Array of posterior emissions
    ndarray : 2d, n_months/freq_months x n_box
        Array of posterior emissions uncertainties (1-sigma)
    """

    emissions = emissions_prior.copy()
    emissions_sd = np.zeros_like(emissions)
    for ti in range(int(emissions.shape[0]/freq_months)):
        for bi in range(4):
            if from_zero:
                # x is emissions
                emissions[ti*freq_months:(ti+1)*freq_months, bi] = x_hat[4*ti + bi]
            else:
                # x is deviation from prior
                emissions[ti*freq_months:(ti+1)*freq_months, bi] += x_hat[4*ti + bi]
            if calculate_uncertainty == True:
                emissions_sd[ti*freq_months:(ti+1)*freq_months, bi] = np.sqrt(P_hat[4*ti + bi, 4*ti + bi])

    return emissions, emissions_sd


def posterior_mf(mf_prior, sensitivity, x_hat, P_hat,
                calc_uncertainty=True, from_zero=False):
    """Calculate linear adjustment to mole fraction array

    Parameters
    ----------
    mf_prior : ndarray
        A priori mole fractions
    sensitivity : ndarray
        Sensitivity matrix (not filtered for missing observations)
    x_hat : ndarray
        Posterior adjustment to emissions (Gg/yr)
    P_hat : ndarray
        Uncertainty covariance in posterior emissions (Gg/yr)
    calc_uncertainty : Boolean, default=True
        Set to false to return zeros instead of calculating uncertainty
    
    Returns 
    -------
    ndarray : 2d, n_months x n_box
        Array of posterior mole fractions
    ndarray : 2d, n_months x n_box
        Array of mole fraction uncertainties (1-sigma)
    """

    def mf_reshape(y):
        return y.reshape(int(len(y)/4), 4)

    # Note that this is labelled y_posterior, but dimension is larger than y
    #  because of obs that have been removed from y (sensitivity is used, rather than H)
    y_posterior = sensitivity @ x_hat
    if calc_uncertainty:
        R_posterior = sensitivity @ P_hat @ sensitivity.T
    else:
        R_posterior = np.zeros((len(y_posterior), len(y_posterior)))

    # Return arrays reshaped to match mf format (time, box)
    if from_zero:
        return mf_reshape(y_posterior), mf_reshape(np.sqrt(np.diag(R_posterior)))
    else:
        return mf_reshape(y_posterior) + mf_prior[:, :4], mf_reshape(np.sqrt(np.diag(R_posterior)))


def calc_lifetime_uncertainty(lifetime_fractional_error, steady_state_lifetime, burden, emissions):
    """Lifetime uncertainty function following Rigby et al., 2014

    Parameters
    ----------
    lifetime_fractional_error : float
        Relative uncertainty in 1/lifetime
    steady_state_lifetime : float
        Steady state lifetime in years
    burden : ndarray
        Burden in each atmospheric box at each timestep
    emissions : ndarray
        Emissions in four surface boxes at each timestep

    Returns
    -------
    ndarray
        Lifetime uncertainty function (Gg/yr)
    """

    global_burden = burden.sum(axis=1)

    # Scale the burden by absolute emissions in each box
    global_emissions = np.abs(emissions).sum(axis=1)
    burden_scaled = np.vstack([global_burden * np.abs(emissions[:, bi])/global_emissions for bi in range(4)]).T

    #TODO: Make unit conversion more robust throughout. Does this work if emissions are in Tg/yr?
    lifetime_uncertainty = burden_scaled * lifetime_fractional_error / steady_state_lifetime / 1e9

    return lifetime_uncertainty


def difference_operator(nx, freq):
    """Calculate differencing operator

    Differences are calculated between quantities separated by one year
    Hence frequency term is required, which tells us the number of 
    elements between years

    Parameters
    ----------
    nx : int
        Number of elements in state vector
    freq : int
        Number of state vector elements between years
    """

    D = np.zeros((nx, nx))

    for bi in range(4):
        for yi in range(int(nx/freq/4)-1):
            for fi in range(freq):
                xi = 4*(yi*freq + fi) + bi
                D[xi, xi] = -1.
                D[xi, xi + 4*freq] = 1.
    
    return D


class Inverse_method:
    """Inverse method modules

    Each method must have a corresponding {method}_posterior function that allows
    posterior mole fraction and emissions matrices to be calculated

    Each method must also have a corresponding {method}_posterior_ensemble function
    that calculates an ensemble of model outputs (potentially including systematic
    uncertainties)
    """

    def analytical_gaussian(self):
        """
        TODO: Update this docstring
        Perform a Gaussian analytical inversion assuming linearity.
        The inferred value 'x_hat' is the difference in emissions from the reference run.
        
            Parameters:
                y (array)         : Deviation from a priori emissions
                H (array)         : Sensitivity matrix
                x_a (array)       : A priori deviation (defaults to zeros)
                R (square matrix) : Model-measurement covariance matrix
                P_inv (square matrix) : Emissions inverse covariance matrix
            Returns:
                x_hat (array): Posterior mean difference from a priori
                P_hat (array): Posterior covariance matrix
        """
        R_inv = np.linalg.inv(self.mat.R)
        self.mat.x_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv) @ \
                        (self.mat.H.T @ R_inv @ self.mat.y + self.mat.P_inv @ self.mat.x_a)
        self.mat.P_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv)
    
    
    def analytical_gaussian_posterior(self, from_zero=False):
        """Method to process posterior mole fractions and emissions
        """

        # Posterior emissions
        if from_zero:
            x_emissions = self.mat.x_hat[1:]
            P_emissions = self.mat.P_hat[1:,1:]
        else:
            x_emissions = self.mat.x_hat
            P_emissions = self.mat.P_hat
        
        emissions, emissions_sd = posterior_emissions(self.mod_prior.emissions,
                                                        self.sensitivity.freq_months,
                                                        x_emissions,
                                                        P_emissions,
                                                        from_zero=from_zero)

        # Posterior mole fraction
        mf, mf_sd = posterior_mf(self.mod_prior.mf,
                                    self.sensitivity.sensitivity,
                                    self.mat.x_hat,
                                    self.mat.P_hat, from_zero=from_zero)

        # Rerun forward model with optimized emissions
        self.mod.emissions = emissions.copy()
        if from_zero:
            self.mod.ic[:] = self.mat.x_hat[0]
        self.mod.run(verbose=False)

        # Check that mole fraction agrees with sensitivity*emissions
        if not np.allclose(self.mod.mf[:, :4], mf, rtol=0.001):
            raise Exception("Optimized model run doesn't match linear prediction")

        # Store posterior model
        self.mod_posterior = Store_model(self.mod)

        # Add in analytical uncertainties
        self.mod_posterior.emissions_sd = emissions_sd
        self.mod_posterior.mf_sd = mf_sd


    def analytical_gaussian_posterior_ensemble(self,
                                                n_sample=1000,
                                                scale_error=0.,
                                                lifetime_error=0.,
                                                transport_error=0.01):
        """Create an ensemble of posterior emissions and mole fractions

        Parameters
        ----------
        n_sample : int, optional
            Number of samples of posterior distribution, by default 1000
        scale_error : float, optional
            Fractional scale uncertainty (e.g., 0.02 for 2%), by default 0.
        lifetime_error : float, optional
            Fractional uncertainty in 1/lifetime, by default 0.
        transport_error : float, optional
            Fractional uncertainty due to model transport, by default 0.01

        Returns
        -------
        ndarray (n_months, n_boxes, n_samples)
            Posterior emissions ensemble
        ndarray (n_months, n_boxes, n_samples)
            Posterior mole fraction ensemble
        """

        emissions_ensemble = np.zeros(self.mod_prior.emissions.shape + (n_sample,))

        L_hat = np.linalg.cholesky(self.mat.P_hat)

        emissions_lifetime_uncertainty = calc_lifetime_uncertainty(lifetime_error,
                                                                self.mod.steady_state_lifetime,
                                                                self.mod_posterior.burden,
                                                                self.mod_posterior.emissions)

        emissions_ensemble = []
        mf_ensemble = []

        for i in range(n_sample):
            x_sample = self.mat.x_hat + L_hat @ np.random.normal(size=self.mat.x_hat.size)

            mf_sample, _ = posterior_mf(self.mod_prior.mf,
                                        self.sensitivity.sensitivity,
                                        x_sample,
                                        self.mat.P_hat,
                                        calc_uncertainty=False)
            
            emissions_sample, _ = posterior_emissions(self.mod_prior.emissions,
                                                    self.sensitivity.freq_months,
                                                    x_sample,
                                                    self.mat.P_hat)

            # Scale uncertainty
            mf_sample *= (1. + np.random.normal() * scale_error)
            emissions_sample *= (1. + np.random.normal() * scale_error)

            # Transport uncertainty
            emissions_sample *= (1. + np.random.normal() * transport_error)

            # Lifetime uncertainty
            emissions_sample += np.random.normal() * emissions_lifetime_uncertainty

            emissions_ensemble.append(emissions_sample)
            mf_ensemble.append(mf_sample)

        return np.dstack(emissions_ensemble), np.dstack(mf_ensemble)

    
    def rigby14(self):
        """Emissions growth-constrainted method of Rigby et al., 2011 and 2014
        """
        
        # Difference operator
        D = difference_operator(len(self.mat.x_a), int(12/self.sensitivity.freq_months))

        H = self.mat.H.copy()

        R_inv = np.linalg.inv(self.mat.R)
        self.mat.P_hat = np.linalg.inv(H.T @ R_inv @ H + D.T @ self.mat.P_inv @ D)
        self.mat.x_hat = self.mat.P_hat @ (H.T @ R_inv @ self.mat.y + D.T @ self.mat.P_inv @ self.mat.x_a)
        
        
    def rigby14_posterior(self):
        """
        The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()


    def rigby14_posterior_ensemble(self, **kwargs):
        """
        The same as a standard analytical Gaussian
        """

        return self.analytical_gaussian_posterior_ensemble(**kwargs)


    def iterative_rigby14(self):
        """
        As rigby14 but optimises x_hat such that all values are >=0. 
        Assumes linearity in the uncertainty.
        """
        H = self.mat.H.copy()
        R_inv = np.linalg.inv(self.mat.R)
        # Difference operator
        D = difference_operator(len(self.mat.x_a), int(12/self.sensitivity.freq_months))
        #Lower limit
        freq_months = self.sensitivity.freq_months
        apriori = self.mod_prior.emissions.copy()
        llim = np.zeros(int(len(apriori.ravel())/(freq_months)))
        for bi in range(4):
            llim[bi::4] = -1*np.mean(apriori[:,bi].reshape(-1, freq_months), axis=1)
        bounds = tuple((ll, None) for ll in llim)
        #Cost function to minimise
        def cost(x):
            return (self.mat.y-H@x).T @ R_inv @ (self.mat.y-H@x) + (D@x-self.mat.x_a).T @ self.mat.P_inv @ (D@x-self.mat.x_a)
        cst = 1e18
        #Initialise with 5 different starting values to avoid local minima
        for i in range(5):
            xout0 = minimize(cost, llim-llim*np.random.random(size=len(llim)), method='L-BFGS-B', 
                            bounds=bounds)
            if xout0.fun < cst:
                cst = xout0.fun
                xout = xout0
        self.mat.x_hat = xout.x
        self.mat.P_hat = np.linalg.inv(H.T @ R_inv @ H + D.T @ self.mat.P_inv @ D)

    def iterative_rigby14_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        

    def iterative_rigby14_posterior_ensemble(self, **kwargs):
        """
        The same as a standard analytical Gaussian
        """

        return self.analytical_gaussian_posterior_ensemble(**kwargs)


    def empirical_bayes(self):
        """
        As iterative_rigby14 but optimises an iid model uncertainty. 
        i.e. there is an additional model error added to the measurement error, which is 
        assumed to be the same at all times and in all boxes.
        It currently also assumes that the R matrix is diagonal to speed up having to do a matrix
        inversion at each step.
        """
        print("Caution: R matrix is assumed to be diagonal.")
        H = self.mat.H.copy()
        R = self.mat.R
        # Difference operator
        D = difference_operator(len(self.mat.x_a), int(12/self.sensitivity.freq_months))
        #Lower limit
        freq_months = self.sensitivity.freq_months
        apriori = self.mod_prior.emissions.copy()
        llim = np.zeros(int(len(apriori.ravel())/(freq_months)))
        for bi in range(4):
            llim[bi::4] = -1*np.mean(apriori[:,bi].reshape(-1, freq_months), axis=1)
        llim = np.append(llim,[1e-20])
        bounds = tuple((ll, None) for ll in llim)
        #Cost function to minimise
        def cost(x):
            R_inv =1./(np.diag(R) + x[-1])
            return np.sum((self.mat.y-H@x[:-1])**2 / R_inv) + \
                   (D@x[:-1]-self.mat.x_a).T @ self.mat.P_inv @ (D@x[:-1]-self.mat.x_a) + \
                   np.sum(np.log(R_inv))
        cst = 1e18
        #Initialise with 5 different starting values to avoid local minima
        for i in range(5):
            init0 = llim-llim*np.random.random(size=len(llim))
            init0[-1] = np.random.random()*2.
            xout0 = minimize(cost, init0, method='L-BFGS-B', 
                            bounds=bounds)
            if xout0.fun < cst:
                cst = xout0.fun
                xout = xout0
        self.mat.x_hat = xout.x[:-1]
        self.mat.P_hat = np.linalg.inv(H.T @ np.linalg.inv(R + np.diag(np.repeat(xout.x[-1], ny))) @ H + \
                                       D.T @ self.mat.P_inv @ D)

    def empirical_bayes_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        

    def empirical_bayes_posterior_ensemble(self, **kwargs):
        """
        The same as a standard analytical Gaussian
        """

        return self.analytical_gaussian_posterior_ensemble(**kwargs)


    def mcmc_analytical(self):

        with pm.Model() as model:
            x = pm.MvNormal("x", mu=self.mat.x_a, tau=self.mat.P_inv, shape=(self.mat.x_a.shape[0]))
            y_observed = pm.MvNormal(
                "y",
                mu=self.mat.H @ x,
                cov=self.mat.R,
                observed=self.mat.y,
            )
    
            # trace = pm.sample(20000, tune=10000, chains=2, step=pm.Metropolis(),
            #                 return_inferencedata=True, progressbar=False)
            trace = pm.sample(500, return_inferencedata=True, tune=500)

        self.mat.x_trace = trace.posterior.sel(chain=0).x.data

        # Store x and P to make posterior processing simpler (but don't use this for posterior ensemble)
        self.mat.x_hat = trace.posterior.sel(chain=0).x.mean(dim="draw").data
        residual = trace.posterior.sel(chain=0).x.data - self.mat.x_hat
        self.mat.P_hat = (residual.T @ residual)/self.mat.x_hat.shape[0]
    

    def mcmc_analytical_posterior(self):
        """As an approximation, use same as analytical Gaussian
        """

        self.analytical_gaussian_posterior()


    def mcmc_analytical_posterior_ensemble(self,
                                            n_sample=1000, # This isn't needed... just putting it here for now otherwise an error is thrown later
                                            scale_error=0.,
                                            lifetime_error=0.,
                                            transport_error=0.01,
                                            from_zero=False):

        n_sample = self.mat.x_trace.shape[0]

        emissions_ensemble = np.zeros(self.mod_prior.emissions.shape + (n_sample,))

        emissions_lifetime_uncertainty = calc_lifetime_uncertainty(lifetime_error,
                                                                self.mod.steady_state_lifetime,
                                                                self.mod_posterior.burden,
                                                                self.mod_posterior.emissions)

        emissions_ensemble = []
        mf_ensemble = []

        for i in range(n_sample):
            x_sample = self.mat.x_a + self.mat.x_trace[i, :]

            mf_sample, _ = posterior_mf(self.mod_prior.mf,
                                        self.sensitivity.sensitivity,
                                        x_sample,
                                        self.mat.P_hat,
                                        calc_uncertainty=False,
                                        from_zero=from_zero)
            
            emissions_sample, _ = posterior_emissions(self.mod_prior.emissions,
                                                    self.sensitivity.freq_months,
                                                    x_sample,
                                                    self.mat.P_hat,
                                                    calculate_uncertainty=False,
                                                    from_zero=from_zero)

            # Scale uncertainty
            mf_sample *= (1. + np.random.normal() * scale_error)
            emissions_sample *= (1. + np.random.normal() * scale_error)

            # Transport uncertainty
            emissions_sample *= (1. + np.random.normal() * transport_error)

            # Lifetime uncertainty
            emissions_sample += np.random.normal() * emissions_lifetime_uncertainty

            emissions_ensemble.append(emissions_sample)
            mf_ensemble.append(mf_sample)

        return np.dstack(emissions_ensemble), np.dstack(mf_ensemble)


    def mcmc_lat_gradient(self, global_method="spline"):
        '''Must be run with sensitivity_from_zero
        '''

        def logistic(L, k, x0, x):
            '''Define Logistic function for latitudinal gradient'''
            return L/(1. + np.exp(-k*(x - x0)))

        if self.sensitivity.freq_months != 1:
            raise Exception("Must provide monthly sensitivity")

        nyears = int(self.mod_prior.emissions.shape[0]/12)

        # number of time periods to split latitude gradient into
        # Factor of 12 so that we don't have any remainders
        lat_grad_years = [10, 20, 30, 40, 60, 1000]
        lat_grad_sections = [1,  2,  3,  4,  6,  6]
        # lat_grad_sections = [6,  12,  24,  36,  48,  48]

        lat_grad_n = lat_grad_sections[bisect_right(lat_grad_years, nyears)]

        if global_method == "annual":

            emissions_global = self.mod_prior.emissions.sum(axis=1)
            emissions_global_annual = emissions_global.reshape((int(emissions_global.shape[0]/12), 12)).mean(axis=1)

        elif global_method == "spline":

            if nyears < 10:
                raise Exception("Currently, recommend at least 10 years for spinup")

            spinup = 10 # years
            num_knots = int((nyears-spinup)/2)

            finite_times = np.isfinite(self.obs.mf) * \
                    np.repeat(np.expand_dims(self.mod_prior.time, axis=1), 4, axis=1)

            # distribute knots over period after spinup, weighted by data density
            knots_list = np.quantile(finite_times[(finite_times > 0.) * \
                    (finite_times > self.obs.time[0] + spinup)], np.linspace(0, 1, num_knots))

            # prepend start year
            knots_list = np.insert(knots_list, 0, self.obs.time[0])

            # B-spline design matrix
            B = np.asarray(dmatrix(
                    "bs(time, knots=knots, degree=3, include_intercept=True) - 1",
                    {"time": self.mod_prior.time, "knots": knots_list[1:-1]},
                    ))

        with pm.Model() as model:

            # # Logistic function parameters (L is normalisation so that sum from 0 -> 3 is 1)
            # k = pm.Uniform("k", lower=0.1, upper=3., shape=(lat_grad_n))
            # x0 = pm.Uniform("x0", lower=0., upper=10., shape=(lat_grad_n))
            # L = 1/(1/(1+np.exp(-k*(0 - x0))) + \
            #         1/(1+np.exp(-k*(1 - x0))) + \
            #         1/(1+np.exp(-k*(2 - x0))) + \
            #         1/(1+np.exp(-k*(3 - x0))))

            # # Get scaling factor for each box (note that these are intentionally ordered 3 -> 0, because box0 is the biggest)
            # x_box3 = pm.Deterministic("x_box3", logistic(L, k, x0, 0))
            # x_box2 = pm.Deterministic("x_box2", logistic(L, k, x0, 1))
            # x_box1 = pm.Deterministic("x_box1", logistic(L, k, x0, 2))
            # x_box0 = pm.Deterministic("x_box0", logistic(L, k, x0, 3))

            if global_method == "annual":

                x_global_annual = pm.TruncatedNormal("x_global",
                                                    mu=emissions_global_annual,
                                                    sigma=emissions_global_annual,
                                                    shape=(nyears,),
                                                    lower=np.zeros(nyears)
                                                    )

                x_global_monthly = at.repeat(x_global_annual, 12)

            elif global_method == "spline":

                # Spline weights
                x_knots3 = pm.TruncatedNormal("x_knots3",
                                            mu=1, sigma=1, lower=0, 
                                            shape = B.shape[1],
                                            )
                x_knots2 = pm.TruncatedNormal("x_knots2",
                                            mu=1, sigma=1, lower=x_knots3, 
                                            shape = B.shape[1],
                                            )
                x_knots1 = pm.TruncatedNormal("x_knots1",
                                            mu=1, sigma=1, lower=x_knots2, 
                                            shape = B.shape[1],
                                            )
                x_knots0 = pm.TruncatedNormal("x_knots0",
                                            mu=1, sigma=1, lower=x_knots1, 
                                            shape = B.shape[1],
                                            )

                # Single global scaling factor
                x_global = pm.TruncatedNormal("x_global", mu=100., sigma=100., lower=0.)

                # x_global_monthly = pm.Deterministic("x_global_monthly", x_global * \
                #                                     pm.math.dot(B, x_knots))

            # x_boxes = at.stack([at.repeat(x_box0, nyears*12/lat_grad_n) * x_global_monthly,
            #                     at.repeat(x_box1, nyears*12/lat_grad_n) * x_global_monthly,
            #                     at.repeat(x_box2, nyears*12/lat_grad_n) * x_global_monthly,
            #                     at.repeat(x_box3, nyears*12/lat_grad_n) * x_global_monthly], axis=1)

            x_boxes = pm.Deterministic("x_boxes", x_global * at.stack([pm.math.dot(B, x_knots0),
                                                                        pm.math.dot(B, x_knots1),
                                                                        pm.math.dot(B, x_knots2),
                                                                        pm.math.dot(B, x_knots3)], axis=1
                                                                        ))

            x_emissions = pm.Deterministic("x_emissions", at.flatten(x_boxes))

            x_ic = pm.Normal("x_ic",
                        mu=self.mod_prior.ic[0],
                        sigma=self.mod_prior.ic[0]*0.1,
                        shape=(1,)) # TODO: currently hard-wired 10% uncertainty

            x = pm.Deterministic("x", at.concatenate([x_ic, x_emissions]))

            # Different model uncertainty for each site and instrument
            site_instrument = np.unique(self.mat.y_site_instrument)
            model_error = pm.HalfNormal("model_error",
                                        sigma=np.mean(np.sqrt(np.diag(self.mat.R))),
                                        shape=len(site_instrument))
            
            y_model_error = at.zeros_like(self.mat.y)
            for i, si in enumerate(site_instrument):
                indices = np.asarray(self.mat.y_site_instrument == si).nonzero()
                y_model_error = at.set_subtensor(y_model_error[indices], model_error[i])

            y_sigma = pm.Deterministic("y_sigma", np.sqrt(np.diag(self.mat.R)) + y_model_error)

            y_observed = pm.Normal(
                "y",
                mu=self.mat.H @ x,
                sigma=y_sigma,
                observed=self.mat.y,
            )

            #prior = pm.sample_prior_predictive(samples=10, model=model)

            # trace = pm.sample(return_inferencedata=True)
            trace = pm.sample(draws=500, tune=100, 
                            return_inferencedata=True,
                            step=pm.Metropolis())

        self.mat.trace = trace.copy()
        #self.mat.prior = prior.copy()

        self.mat.x_trace = trace.posterior.sel(chain=0).x.data

        # Store x and P to make posterior processing simpler (but don't use this for posterior ensemble)
        self.mat.x_hat = trace.posterior.sel(chain=0).x.mean(dim="draw").data
        residual = trace.posterior.sel(chain=0).x.data - self.mat.x_hat
        self.mat.P_hat = (residual.T @ residual)/self.mat.x_hat.shape[0]


    def mcmc_lat_gradient_posterior(self):
        self.analytical_gaussian_posterior(from_zero=True)


    def mcmc_lat_gradient_posterior_ensemble(self, **kwargs):
        """
        The same as a standard analytical Gaussian
        """

        return self.mcmc_analytical_posterior_ensemble(**kwargs, from_zero=True)