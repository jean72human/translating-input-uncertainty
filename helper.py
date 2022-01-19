import numpy as np

## MELD = 3.78×ln[serum bilirubin (mg/dL)] + 11.2×ln[INR] + 9.57×ln[serum creatinine (mg/dL)] + 6.43
## MELD-Na = MELD + 1.32 * (137 – sodium mmol/L) – [0.033 * MELD * (137 – sodium mmol/L)]
def meld_score(sb,inr,sc,na):
    na = min(137,na)
    meld = 3.78 * np.log(max(1,sb)) + 11.2 * np.log(max(1,inr)) + 9.57 * np.log(min(4,max(1,sc))) + 6.43
    meld_na = meld + 1.32 * (137 - na) - (0.033 * meld * (137 - na))
    return meld_na

def get_sb(healthy=True,use_mean=False): 
    mean, var = (1.1, 0.26) if healthy else (2.7, 0.26) # use a mean of 1.1 and std of 0.26 for healthy data else use a mean of 2.7 and same std
    if use_mean: return mean
    return np.random.normal(mean,var)

def get_inr(healthy=True,use_mean=False):
    mean, var =(1,0.07) if healthy else (1.4,0.07)
    if use_mean: return mean
    return np.random.normal(mean,var)  

def get_sc(healthy=True,use_mean=False):
    mean, var = (1,0.1) if healthy else (1.6,0.1)
    if use_mean: return mean
    return np.random.normal(mean,var)

def get_na(healthy=True,use_mean=False):
    mean, var = (140,1.7) if healthy else (130,1.7)
    if use_mean: return mean
    return np.random.normal(mean,var)