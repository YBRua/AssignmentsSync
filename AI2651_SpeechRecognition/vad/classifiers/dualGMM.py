from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import numpy as np
import scipy.signal as signal


class DualGMMClassifier():
    """Dual GMM Classifier.

    Models voiced and unvoiced frames with a GMM respectively.
    """
    def __init__(
        self, n_components=3,
        covariance_type='full',
        max_iter=500,
        verbose=0,
        random_state=None,
    ):
        self.voiced_gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
        )
        self.unvoiced_gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
        )

    def fit(self, X, Y):
        """Fit the classifier.

        Arugments:
            X: 2darray -- (n_samples, n_features array).
            Y: 1darray -- labels.

        Returns:
            self
        """
        X_voiced = X[Y == 1]
        X_unvoiced = X[Y == 0]
        return self._fit(X_voiced, X_unvoiced, Y)

    def _fit(self, X_voiced, X_unvoiced, Y_train):
        """Fit the Gaussian Mixtures of the classifier.

        Arguments:
            X_voiced: 2darray -- (n_samples, n_features) array.
            X_unvoiced: 2darray -- (n_samples, n_features) array.
            Y_train: 1darray -- labels of each frame.

        Returns:
            self -- Classifier instance.
        """
        self.voiced_gmm = self.voiced_gmm.fit(X_voiced)
        self.unvoiced_gmm = self.unvoiced_gmm.fit(X_unvoiced)
        self._compute_prior(Y_train)

        return self

    def _compute_prior(self, Y_train):
        self.voiced_prior = np.sum(Y_train) / Y_train.shape[0]
        self.unvoiced_prior = 1 - self.voiced_prior

    def predict_log_proba(self, X):
        """Predicts the log posterior probability of voiced and unvoiced,
        given an input X.

        Arguments:
            X: 2darray -- (n_samples, n_features) array of input.

        Returns:
            [voiced, unvoiced] log posteriors.
        """
        voiced_log_likelihood = np.log(self.voiced_prior) \
            + self._compute_voiced_log_likelihood(X)
        unvoiced_log_likelihood = np.log(self.unvoiced_prior) \
            + self._compute_unvoiced_log_likelihood(X)
        log_evidence = logsumexp(
            [voiced_log_likelihood, unvoiced_log_likelihood],
            axis=0
        )
        voiced_log_prob = voiced_log_likelihood - log_evidence
        unvoiced_log_prob = unvoiced_log_likelihood - log_evidence

        return np.concatenate(
            [voiced_log_prob, unvoiced_log_prob],
            axis=1
        )

    def predict_proba(self, X):
        """Predicts the posterior probability, given an input X.

        Arguments:
            X: 2darray -- (n_samples, n_features) input.

        Returns:
            [voiced, unvoiced] probability.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """Pedicts the label of each frame of input X.

        Arguments:
            X: 2darray -- (n_samples, n_features) input.

        Returns:
            (n_samples) {0, 1} labels.
        """
        proba = self.predict_proba(X)
        proba = proba[:, 0]
        return np.where(proba >= 0.5, 1, 0)

    def predict_smoothed_proba(self, x):
        """Uses a L=15 mean filter to smooth the probability output.
        """
        proba = self.predict_proba(x)[:, 0]
        filter = np.full(15, 1/15)
        return signal.convolve(proba, filter, mode='same')

    def predict_smoothed(self, x):
        smoothed_proba = self.predict_smoothed_proba(x)
        return np.where(smoothed_proba >= 0.5, 1, 0)

    def _compute_voiced_log_likelihood(self, X):
        log_likelihood, _ = self.voiced_gmm._estimate_log_prob_resp(X)

        return np.expand_dims(log_likelihood, axis=1)

    def _compute_unvoiced_log_likelihood(self, X):
        log_likelihood, _ = self.unvoiced_gmm._estimate_log_prob_resp(X)

        return np.expand_dims(log_likelihood, axis=1)
