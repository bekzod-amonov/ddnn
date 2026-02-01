# load packages
import locale
import os
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import calendar
import torch
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any


# Tutor's code
def DST_trafo(X, Xtime, tz="CET"):
    """Converts a time series DataFrame to a DST-adjusted array

    The function takes a DataFrame of D*S rows and N columns and returns
    an array of shape (D,S,N) where D is the number of days, S the number
    of observations per day and N the number of variables. The function deals
    with the DST problem by averaging the additional hour in October and
    interpolating the missing hour in March.

    Parameters
    ----------
    X : DataFrame
        The time series DataFrame of shape (D*S,N) to be DST-adjusted.
    Xtime : datetime Series
        The series of length D*S containing UTC dates corresponding to the
        DataFrame X.
    tz : str
        The timezone to which the data needs to be adjusted to. The current
        implementation was not tested with other timezones than CET.

    Returns
    -------
    ndarray
        an ndarray of DST-adjusted variables of shape (D,S,N).
    """
    Xinit = X.values
    if len(Xinit.shape) == 1:
        Xinit = np.reshape(Xinit, (len(Xinit), 1))
    atime_init = Xtime.dt.tz_convert('UTC').astype('int64') # <- added this part
    #atime_init = pd.to_numeric(Xtime)
    freq = atime_init.diff().value_counts().idxmax()
    S = int(24*60*60 * 10**9 / freq)
    atime = pd.DataFrame(
        np.arange(start=atime_init.iloc[0], stop=atime_init.iloc[-1]+freq,
                  step=freq))
    idmatch = atime.reset_index().set_index(0).loc[atime_init, "index"].values
    X = np.empty((len(atime), Xinit.shape[1]))
    X[:] = np.nan
    X[idmatch] = Xinit

    new_time = Xtime.dt.tz_convert(tz).reset_index(drop=True)
    DLf = new_time.dt.strftime("%Y-%m-%d").unique()
    days = pd.Series(pd.to_datetime(DLf))

    # EUROPE
    DST_SPRING = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        30) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST_FALL = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        100) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST = ~(DST_SPRING | DST_FALL)

    time_start = new_time.iloc[range(
        S+int(S/24))].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    time_end = new_time.iloc[range(-S-int(S/24), 0)
                             ].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    Dlen = len(DLf)
    Shift = 2  # for CET

    X_dim = X.shape[1]

    Xout = np.empty((Dlen, S, X_dim))
    Xout[:] = np.nan

    k = 0
    # first entry:
    i_d = 0
    idx = time_start[time_start.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, S-1-idx[::-1], ] = X[range(k, len(idx)+k), ]
    elif DST_SPRING[i_d]:
        tmp = S-1-idx[::-1]
        # MARCH
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] == Shift * S/24-1:
                Xout[i_d, range(int(S-S/24 - len(idx) + i_S+1),
                                int(S-S/24 - len(idx) + i_S+1 + S/24)),
                     ] = X[[k+i_S, ]] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if tmp[i_S] > Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S+S/24), ] = X[k+i_S, ]
    else:
        tmp = S-idx[::-1]
        # OCTOBER
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, int(S+S/24 - len(idx)+i_S), ] = 0.5 * \
                    (X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if tmp[i_S] > (Shift+2) * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S-S/24), ] = X[k+i_S, ]
    k += len(idx)
    for i_d in range(1, len(DLf)-1):
        if DST[i_d]:
            idx = S
            Xout[i_d, range(idx), ] = X[range(k, k+idx), ]
        elif DST_SPRING[i_d]:
            idx = int(S-S/24)
            # MARCH
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S == Shift * S/24-1:
                    Xout[i_d, range(int(i_S+1),
                                    int(i_S + 1 + S/24)),
                         ] = X[[k+i_S, ]] + np.transpose(
                        np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                            S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
                if i_S > Shift * S/24-1:
                    Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
        else:
            idx = int(S+S/24)
            # October
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                    Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
                if i_S > (Shift+2) * S/24-1:
                    Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
        k += idx
    # last
    i_d = len(DLf)-1
    idx = time_end[time_end.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, range(len(idx)), ] = X[range(k, k+len(idx)), ]
    elif DST_SPRING[i_d]:
        # MARCH
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S == Shift * S/24-1:
                Xout[i_d, range(int(i_S+1),
                                int(i_S + 1 + S/24)), ] = X[[k+i_S, ]
                                                            ] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if i_S > Shift * S/24-1:
                Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
    else:
        # OCTOBER
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if i_S > (Shift+2) * S/24-1:
                Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
    return Xout
  
  
def prepare_dataset_tensor_modified(
    csv_path: Union[str, Path],
    tz: str,
    seed: int,
    test_days: int,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, pd.Series, torch.Tensor]:

    # Deterministic environment & device
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(Path(csv_path))
    time_utc = pd.to_datetime(df["time_utc"], utc=True, format="%Y-%m-%d %H:%M:%S")

    time_lt = time_utc.dt.tz_convert(tz)
    data_array = DST_trafo(X=df.iloc[:, 1:], Xtime=time_utc, tz=tz)
    data_tensor = torch.tensor(data_array, dtype=dtype, device=device)
    price_tensor = data_tensor[..., 0]

    if test_days >= data_tensor.shape[0]:
        raise ValueError("test_days must be smaller than the dataset length")
    
    train_tensor = data_tensor # data_tensor[:-test_days]

    # Build local‑time date index parallel to tensor rows
    local_dates = pd.Series(time_lt.dt.date.unique())
    train_dates = local_dates # local_dates[:-test_days]

    return data_tensor, train_tensor, train_dates, price_tensor