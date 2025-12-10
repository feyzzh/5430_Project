import numpy as np
import pandas as pd
import math

class FitHMM:
    def __init__(self, X, full_df, K=3):
        """
        Assumes X is a dataframe with the factors you want as predictors in the columns
        and the Date(time) is the index
        """
        self.full_df = full_df
        self.data = X
        self.X = X.dropna().values
        self.aligned_dates = X.dropna().index
        self.get_norm_stats(self.X)
        self.K = K


    def get_norm_stats(self, X):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0, ddof=1)
        self.X_stdized = (X - self.X_mean) / self.X_std

    def _build_regime_segments(self, df_with_regimes):
        """
        Compress consecutive rows with same Regime into segments:
        [
          {"regime": "bull", "start": "...", "end": "..."},
          ...
        ]
        """
        df = df_with_regimes.sort_index()
        regimes = df['Regime']

        segments = []
        current_regime = None
        current_start = None
        prev_ts = None

        print(regimes)

        for ts, reg in regimes.items():
            if pd.isna(reg):
                if current_regime is not None:
                    segments.append({
                        "regime": current_regime,
                        "start": current_start.isoformat(),
                        "end": prev_ts.isoformat(),
                    })
                    current_regime = None
                    current_start = None
            else:
                if reg != current_regime:
                    if current_regime is not None:
                        segments.append({
                            "regime": current_regime,
                            "start": current_start.isoformat(),
                            "end": prev_ts.isoformat(),
                        })
                    current_regime = reg
                    current_start = ts
            prev_ts = ts

        if current_regime is not None and prev_ts is not None:
            segments.append({
                "regime": current_regime,
                "start": current_start.isoformat(),
                "end": prev_ts.isoformat(),
            })

        return segments

    def _label_regimes_from_means(self, means_original,
                              low_vol_threshold=0.2,
                              high_vol_threshold=0.4):
        """
        Heuristic:
        - ret = means_original[k, 0]
        - rng = means_original[k, 2]
        """
        regime_labels = []
        for k in range(means_original.shape[0]):
            ret = means_original[k, 0]   # mean log return
            rng = means_original[k, 2]   # mean range

            if ret > 0 and rng < low_vol_threshold:
                regime_labels.append('bull')
            elif ret < 0 and rng > high_vol_threshold:
                regime_labels.append('bear')
            else:
                regime_labels.append('rebound')
        return regime_labels

    
    def pipeline(self):
        pi, A, means, covs, log_liks = fit_hmm(
            X=self.X_stdized,
            K=self.K,
            max_iter=100,
            tol=1e-4,
            random_state=0
        )
        print("Initial state probabilities (pi):")
        print(pi)
        print("\nTransition matrix (A):")
        print(A)
        print("\nState means (in standardized units):")
        print(means)
        print("\nFinal log-likelihood:", log_liks[-1])
        result = {
            'pi': pi,
            'A': A,
            'means': means,
            'log_liks': log_liks,
            'final_log_liks': log_liks[-1]
        }

        means_original = means * self.X_std + self.X_mean
        print("State means in original feature units:")
        print(means_original)

        states = viterbi(
            X=self.X_stdized, pi=pi, A=A, means=means, covs=covs
        )

        # Create a Series of integer regimes aligned with the non-missing rows:
        regimes_int = pd.Series(states, index=self.aligned_dates, name='RegimeIndex')

        # Now create a full-length regime series (same index as self.data)
        regimes_full = pd.Series(np.nan, index=self.data.index, name='RegimeIndex')
        regimes_full.loc[self.aligned_dates] = regimes_int

        # Stick it into a copy of the original data
        df_with_regimes = self.data.copy()
        df_with_regimes['RegimeIndex'] = regimes_full

        regime_labels = self._label_regimes_from_means(means_original)

        # Map ints -> names
        df_with_regimes['Regime'] = df_with_regimes['RegimeIndex'].map(
            lambda x: regime_labels[int(x)] if pd.notna(x) else None
        )

        df_with_regimes.groupby('Regime')[self.data.columns].mean()

        regime_summary = df_with_regimes.groupby('Regime').mean(numeric_only=True)

        # regimes_int = pd.Series(states, index=self.aligned_dates, name='RegimeIndex')

        # regimes_named = regimes_int.copy()
        # regime_labels = self._label_regimes_from_means(means_original)
        # regimes_named = regimes_named.map(lambda x: regime_labels[int(x)] if not pd.isna(x) else None)
        # regimes_named.name = 'Regime'

        # if self.full_df is not None:
        #     df_with_regimes = self.full_df.copy()
        # else:
        #     df_with_regimes = self.features.copy()

        # df_with_regimes = df_with_regimes.join(regimes_int, how='left')
        # df_with_regimes = df_with_regimes.join(regimes_named, how='left')

        # # 5) Regime summary over numeric cols
        # regime_summary = df_with_regimes.groupby('Regime').mean(numeric_only=True)

        # # 6) Build time_series for frontend (OHLC + logReturn + regime)
        # time_series = []
        # for idx, row in df_with_regimes.iterrows():
        #     def safe_float(x):
        #         try:
        #             if x is None:
        #                 return None
        #             x = float(x)
        #             if math.isnan(x) or math.isinf(x):
        #                 return None
        #             return x
        #         except Exception:
        #             return None

        #     regime_val = row.get('Regime')
        #     if isinstance(regime_val, float) and math.isnan(regime_val):
        #         regime_val = None

        #     time_series.append({
        #         "date": idx,
        #         "open": safe_float(row.get("Open")),
        #         "high": safe_float(row.get("High")),
        #         "low": safe_float(row.get("Low")),
        #         "close": safe_float(row.get("Close")),
        #         "volume": safe_float(row.get("Volume")),
        #         "logReturn": safe_float(row.get("Log Returns")),
        #         "regime": regime_val,
        #         "regimeIndex": safe_float(row.get("RegimeIndex")),
        #     })

        # regime_segments = self._build_regime_segments(df_with_regimes)


        # Prepare time-series rows for frontend
        time_series_rows = [
            {
                "date": idx.isoformat(),
                "regimeIndex": row["RegimeIndex"] if not pd.isna(row["RegimeIndex"]) else None,
                "regime": row["Regime"],
                # + any features you care about
            }
            for idx, row in df_with_regimes.iterrows()
        ]

        result = {
            # "ticker": self.ticker,
            "K": self.K,
            "pi": pi.tolist(),
            "A": A.tolist(),
            "state_means_original": means_original.tolist(),
            "regime_labels": regime_labels,
            "log_likelihood_trace": log_liks.tolist(),
            "regime_summary": regime_summary.to_dict(orient="index"),
            "time_series": time_series_rows,
            # "time_series": time_series,
            # "regime_segments": regime_segments,
        }

        return result, df_with_regimes


def logsumexp(log_vals, axis=None):
    """
    Stable log-sum-exp: log(sum(exp(log_vals))) along given axis.
    """
    m = np.max(log_vals, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(log_vals - m), axis=axis, keepdims=True))

def gaussian_logpdf(x, mean, cov):
    """
    x: (T, D)
    mean: (D,)
    cov: (D, D) positive definite
    Returns: log p(x_t | state) for each t, shape (T,)
    """
    x = np.atleast_2d(x)
    D = x.shape[1]
    mean = mean.reshape(1, D)
    cov = np.atleast_2d(cov)
    
    # Precompute
    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance not positive definite")

    diff = x - mean
    # Mahalanobis term
    mah = np.sum(diff @ inv_cov * diff, axis=1)  # shape (T,)
    
    return -0.5 * (D * np.log(2 * np.pi) + logdet + mah)

def forward_backward_log(X, pi, A, means, covs):
    """
    X: (T, D)
    pi: (K,)
    A: (K, K)
    means: (K, D)
    covs:  (K, D, D)
    
    Returns:
        log_alpha: (T, K)
        log_beta:  (T, K)
        log_lik:   scalar log-likelihood
    """
    T, D = X.shape
    K = pi.shape[0]
    
    # Precompute emission log-likelihoods: log B_t(k) = log p(x_t | state=k)
    log_B = np.zeros((T, K))
    for k in range(K):
        log_B[:, k] = gaussian_logpdf(X, means[k], covs[k])
    
    log_pi = np.log(pi)
    log_A = np.log(A)
    
    # Forward pass: log_alpha
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_B[0]
    
    for t in range(1, T):
        # log_alpha[t, j] = log( sum_i exp(log_alpha[t-1, i] + log_A[i, j]) ) + log_B[t, j]
        trans = log_alpha[t-1][:, None] + log_A  # shape (K, K)
        log_alpha[t] = logsumexp(trans, axis=0).ravel() + log_B[t]
    
    # Backward pass: log_beta
    log_beta = np.zeros((T, K))
    log_beta[-1] = 0.0  # log(1)
    
    for t in range(T-2, -1, -1):
        # log_beta[t, i] = log( sum_j A[i,j] * B_{t+1}(j) * beta_{t+1}(j) )
        trans = log_A + (log_B[t+1] + log_beta[t+1])[None, :]  # shape (K, K)
        log_beta[t] = logsumexp(trans, axis=1).ravel()
    
    # Total log-likelihood: log p(X) = logsumexp(log_alpha[T-1])
    log_lik = logsumexp(log_alpha[-1], axis=0).item()
    
    return log_alpha, log_beta, log_lik, log_B

def e_step(X, pi, A, means, covs):
    """
    Runs forward-backward and computes:
      gamma_t(k) = P(Z_t = k | X)
      xi_t(i,j)  = P(Z_t = i, Z_{t+1} = j | X)
    """
    T, D = X.shape
    K = pi.shape[0]
    
    log_alpha, log_beta, log_lik, log_B = forward_backward_log(X, pi, A, means, covs)
    
    # gamma: (T, K)
    log_gamma = log_alpha + log_beta  # unnormalized
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1)  # normalize per t
    gamma = np.exp(log_gamma)
    
    # xi: (T-1, K, K)
    log_A = np.log(A)
    log_xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        # log xi_t(i,j) ∝ log_alpha[t,i] + log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j]
        log_unnorm = (
            log_alpha[t][:, None]
            + log_A
            + (log_B[t+1] + log_beta[t+1])[None, :]
        )  # (K, K)
        log_xi[t] = log_unnorm - logsumexp(log_unnorm)  # normalize
    xi = np.exp(log_xi)
    
    return gamma, xi, log_lik

def m_step(X, gamma, xi):
    """
    X: (T, D)
    gamma: (T, K)
    xi: (T-1, K, K)
    """
    T, D = X.shape
    Tm1 = T - 1
    K = gamma.shape[1]
    
    # Initial state distribution
    pi_new = gamma[0]  # shape (K,)
    
    # Transition matrix
    xi_sum = xi.sum(axis=0)          # (K, K)
    gamma_sum = gamma[:-1].sum(axis=0)  # (K,)
    A_new = xi_sum / gamma_sum[:, None]
    
    # Means and covariances
    means_new = np.zeros((K, D))
    covs_new = np.zeros((K, D, D))
    
    for k in range(K):
        # Weighted mean
        w = gamma[:, k][:, None]  # (T,1)
        denom = w.sum()
        means_new[k] = (w * X).sum(axis=0) / denom
        
        # Weighted covariance
        diff = X - means_new[k]
        cov = (w * diff).T @ diff / denom
        # Regularize slightly in case of numerical issues
        cov += 1e-6 * np.eye(D)
        covs_new[k] = cov
    
    return pi_new, A_new, means_new, covs_new

def fit_hmm(X, K=2, max_iter=100, tol=1e-4, random_state=0):
    """
    Fit a K-state Gaussian HMM to data X using EM.
    
    X: (T, D)
    Returns:
      pi, A, means, covs, log_liks
    """
    rng = np.random.default_rng(random_state)
    T, D = X.shape
    
    # --- Initialization ---
    # Randomly assign states and estimate initial means/covs roughly
    # or use k-means if you want nicer init.
    gamma_init = rng.random((T, K))
    gamma_init /= gamma_init.sum(axis=1, keepdims=True)
    
    pi = gamma_init[0]
    pi /= pi.sum()
    
    # Uniform transition matrix to start
    A = np.ones((K, K)) / K
    
    # Means: pick random rows of X
    means = X[rng.choice(T, size=K, replace=False)]
    
    # Covs: start as global covariance
    global_cov = np.cov(X.T) if D > 1 else np.array([[np.var(X)]])
    covs = np.array([global_cov.copy() for _ in range(K)])
    
    log_liks = []
    prev_log_lik = -np.inf
    
    for it in range(max_iter):
        # E-step
        gamma, xi, log_lik = e_step(X, pi, A, means, covs)
        log_liks.append(log_lik)
        
        # M-step
        pi, A, means, covs = m_step(X, gamma, xi)
        
        # Convergence check
        if it > 0 and abs(log_lik - prev_log_lik) < tol:
            print(f"Converged at iteration {it}, log-lik = {log_lik:.3f}")
            break
        
        prev_log_lik = log_lik
    
    return pi, A, means, covs, np.array(log_liks)

def viterbi(X, pi, A, means, covs):
    T, D = X.shape
    K = pi.shape[0]
    
    log_pi = np.log(pi)
    log_A = np.log(A)
    
    # Emission log-likelihoods
    log_B = np.zeros((T, K))
    for k in range(K):
        log_B[:, k] = gaussian_logpdf(X, means[k], covs[k])
    
    # DP arrays
    delta = np.zeros((T, K))  # best log-prob up to t, ending in k
    psi = np.zeros((T, K), dtype=int)  # argmax pointers
    
    delta[0] = log_pi + log_B[0]
    
    for t in range(1, T):
        for j in range(K):
            scores = delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_B[t, j]
    
    # Backtrack
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    
    return states

