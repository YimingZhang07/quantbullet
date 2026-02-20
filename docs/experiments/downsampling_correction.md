# Correction Methods for Downsampled Data

When we downsample the non-events (majority class) to match a target event rate, the predicted probabilities from the model trained on this downsampled data will be biased upwards. We need to apply a correction to recover the true probabilities. Let:

- $\pi$: True event rate in the population (or full data).
- $\tilde{\pi}$: Event rate in the downsampled training data.
- $\tilde{p}$: Predicted probability from the model trained on downsampled data.
- $p$: Corrected predicted probability (estimate of the true probability).

## Method 1: Bayes' Theorem Correction (Prior Correction)

This method directly applies Bayes' theorem to adjust the posterior probabilities based on the change in prior probabilities.

### Exact Formula

The exact Bayesian correction is derived by considering the odds. Let $Odds = \frac{p}{1-p}$. The ratio of the posterior odds is equal to the ratio of the prior odds times the likelihood ratio. Assuming the likelihood ratio is unchanged by downsampling:

$$ \frac{p}{1-p} = \frac{\tilde{p}}{1-\tilde{p}} \cdot \frac{\frac{\pi}{1-\pi}}{\frac{\tilde{\pi}}{1-\tilde{\pi}}} $$

Let the prior odds adjustment factor be $K = \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)}$. Then:

$$ \frac{p}{1-p} = K \cdot \frac{\tilde{p}}{1-\tilde{p}} $$

Solving for $p$, we get the exact correction:

$$ p = \frac{K \tilde{p}}{1 - \tilde{p} + K \tilde{p}} $$

### Approximate Formula (When Event Rate is Small)

If the true event rate $\pi$ is very small, then $1-\pi \approx 1$. If we only downsample the non-events and the resulting $\tilde{\pi}$ is also relatively small (or if we just simplify the ratio), we might approximate the prior odds ratio $K$ simply by the ratio of the event rates:

$$ K \approx \frac{\pi}{\tilde{\pi}} $$

Substituting this into the exact formula gives the approximate correction commonly used:

$$ p \approx \frac{\frac{\pi}{\tilde{\pi}} \tilde{p}}{1 + \left(\frac{\pi}{\tilde{\pi}} - 1\right) \tilde{p}} $$

*Note: In the notebook code, this approximate version is implemented as `(ratio * p_raw) / (1 + (ratio - 1) * p_raw)` where `ratio = pi_true / pi_down`.*

## Method 2: King & Zeng Intercept Correction (Logistic Shift)

If the model is a logistic regression (or a GAM with a logit link), downsampling the non-events primarily affects the intercept of the model. The slope coefficients remain consistent estimators of the true coefficients.

The model trained on downsampled data predicts the log-odds (logit):

$$ \text{logit}(\tilde{p}) = \ln\left(\frac{\tilde{p}}{1-\tilde{p}}\right) = \tilde{\beta}_0 + \beta_1 x_1 + \dots $$

To correct the probabilities, we shift the intercept by the log of the prior odds ratio:

$$ \text{logit}(p) = \text{logit}(\tilde{p}) + \ln\left( \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} \right) $$

Therefore, the corrected probability is obtained by taking the expit (sigmoid) of the corrected log-odds:

$$ p = \text{sigmoid}\left( \text{logit}(\tilde{p}) + \ln\left( \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} \right) \right) $$

where $\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$.

*Note: In the notebook code, this is implemented exactly as shifting the logit by `kz_correction`.*

## How Method 1 Leads to Method 2

Method 1 (exact) and Method 2 are mathematically equivalent for models that output probabilities. The difference is purely in the functional form used for the calculation.

Starting from the exact Bayesian formulation in Method 1:

$$ \frac{p}{1-p} = \frac{\tilde{p}}{1-\tilde{p}} \cdot \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} $$

Take the natural logarithm of both sides:

$$ \ln\left(\frac{p}{1-p}\right) = \ln\left(\frac{\tilde{p}}{1-\tilde{p}}\right) + \ln\left( \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} \right) $$

Recognize that $\ln\left(\frac{x}{1-x}\right)$ is the logit function:

$$ \text{logit}(p) = \text{logit}(\tilde{p}) + \ln\left( \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} \right) $$

This is exactly the equation for Method 2. To get the final probability $p$, apply the inverse logit (sigmoid) function to both sides:

$$ p = \text{sigmoid}\left( \text{logit}(\tilde{p}) + \ln\left( \frac{\pi (1-\tilde{\pi})}{\tilde{\pi} (1-\pi)} \right) \right) $$

**Conclusion:** The exact prior correction (Method 1) and the King & Zeng intercept correction (Method 2) are algebraically identical. They will produce the exact same corrected probabilities. The approximate form of Method 1 only holds when $1-\pi \approx 1$ and $1-\tilde{\pi} \approx 1$.
