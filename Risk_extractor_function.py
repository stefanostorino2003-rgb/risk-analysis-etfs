import pandas as pd
import numpy as np
import scipy.stats as ss
def Risk_extractor(Dataset):
    
    mean_daily = Dataset.mean()
    mean_annualized = (1+mean_daily)**252 - 1
    
    st_dev_daily = Dataset.std(ddof=0)
    st_dev_ann = st_dev_daily * np.sqrt(252)
    
    #i considered only the deviation of lossess since i know it's the common way in finance
    losses = Dataset[Dataset < 0]
    excess_square = losses**2
    n = len(Dataset)
    semidev_daily = (excess_square.sum()/n)**0.5
    semidev_ann = semidev_daily * np.sqrt(252)

    #instead of caluclating the risk free rate i assumed a 2%
    Sharpe_ratio = (mean_annualized-0.02)/st_dev_ann

    Sortino_ratio = (mean_annualized-0.02)/semidev_ann

    wealth_index = (1+Dataset).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    duration = []
    current = 0

    for d in drawdown:
        if d >= 0:
            current = 0
        else:
            current += 1
        duration.append(current)
    max_drawdown = drawdown.min()

    DD_duration = pd.Series(duration, index=drawdown.index)
    max_duration = DD_duration.max()

    skewness = ((Dataset - mean_daily)**3).mean()/st_dev_daily**3

    kurtosis = ((Dataset - mean_daily)**4).mean()/st_dev_daily**4
    excess_kurtosis = kurtosis -3

    #to compute var and cvar i considered the 5% worst case scenario daily
    var_historic = -np.percentile(Dataset, 5)

    z = - (ss.norm.ppf(0.05))
    
    z_cf = (
        z
        + (z**2 - 1)*skewness/6
        + (z**3 - 3*z)*(kurtosis - 3)/24
        - (2*z**3 - 5*z)*(skewness**2)/36
        )    
    var_gaussian = -(mean_daily - z * st_dev_daily)
    var_Cornish_Fisher = -(mean_daily - z_cf * st_dev_daily)

    Cvar_historic = -Dataset[Dataset<=-var_historic].mean()

    return {
    "Mean": mean_annualized,
    "Std Dev": st_dev_ann,
    "Semi Dev": semidev_ann,
    "Sharpe Ratio": Sharpe_ratio,
    "Sortino Ratio": Sortino_ratio,
    "Max Drawdown": max_drawdown,
    "Drawdown Duration": max_duration,
    "Skewness": skewness,
    "Kurtosis": kurtosis,
    "Excess Kurtosis": excess_kurtosis,
    "Historic VaR (5%)": var_historic,
    "Gaussian VaR (5%)": var_gaussian,
    "Cornish-Fisher VaR (5%)": var_Cornish_Fisher,
    "Historic CVaR (5%)": Cvar_historic
    }














