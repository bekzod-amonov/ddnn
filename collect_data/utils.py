


# load packages
import locale
import os
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import torch
import random
import torch.nn as nn
import torch.optim as optim

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    atime_init = pd.to_numeric(Xtime)
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


def get_pbas(Bindex, period=365.24, dK=365.24/6, order=4):
    """Estimates periodic B-splines to model the annual periodicity

    Parameters
    ----------
    Bindex : array_like of int
        The array of day numbers for which to estimate the B-splines.
    period : float
        The period of B-splines. By default set to 365.24.
    dK : float
        The equidistance distance used to calculate the knots.
    order : int
        The order of the B-splines. 3 indicates quadratic splines, 4 cubic etc.

    Returns
    -------
    ndarray
        an ndarray of estimated B-splines.
    """
    # ord=4 --> cubic splines
    # dK = equidistance distance
    # support will be 1:n
    n = len(Bindex)
    stp = dK
    x = np.arange(1, period)  # must be sorted!
    lb = x[0]
    ub = x[-1]
    knots = np.arange(lb, ub+stp, step=stp)
    degree = order-1
    Aknots = np.concatenate(
        (knots[0] - knots[-1] + knots[-1-degree:-1], knots,
         knots[-1] + knots[1:degree+1] - knots[0]))

    from bspline import Bspline
    bspl = Bspline(Aknots, degree)
    basisInterior = bspl.collmat(x)
    basisInteriorLeft = basisInterior[:, :degree]
    basisInteriorRight = basisInterior[:, -degree:]
    basis = np.column_stack(
        (basisInterior[:, degree:-degree],
         basisInteriorLeft+basisInteriorRight))
    ret = basis[np.array(Bindex % basis.shape[0], dtype="int"), :]
    return ret


def dm_test(error_a, error_b, hmax=1, power=1):
    # as dm_test with alternative == "less"
    loss_a = (np.abs(error_a)**power).sum(1)**(1/power)
    loss_b = (np.abs(error_b)**power).sum(1)**(1/power)
    delta = loss_a - loss_b
    # estimation of the variance
    delta_var = np.var(delta) / delta.shape[0]
    statistic = delta.mean() / np.sqrt(delta_var)
    delta_length = delta.shape[0]
    k = ((delta_length + 1 - 2 * hmax + (hmax / delta_length)
         * (hmax - 1)) / delta_length)**(1 / 2)
    statistic = statistic * k
    p_value = t.cdf(statistic, df=delta_length-1)

    return {"stat": statistic, "p_val": p_value}


def get_cpacf(y, k=1):
    S = y.shape[1]
    n = y.shape[0]
    cpacf = np.full((S, S), np.nan)
    for s in range(S):
        for l in range(S):
            y_s = y[k:n, s]
            y_l_lagged = y[:(n-k), l]
            cpacf[s, l] = np.corrcoef(y_s, y_l_lagged)[0, 1]
    return cpacf


def pcor(y, x, z):
    XREG = np.column_stack((np.ones(z.shape[0]), z))
    model_y = LinearRegression(fit_intercept=False).fit(X=XREG, y=y)
    model_x = LinearRegression(fit_intercept=False).fit(X=XREG, y=x)
    cor = np.corrcoef(y - model_y.predict(XREG),
                      x - model_x.predict(XREG))[0, 1]
    return cor


def hill(data, start=14, end=None, abline_y=None, ci=0.95, ax=None):
    """Hill estimator translation from R package evir::hill

    Plot the Hill estimate of the tail index of heavy-tailed data, or of an 
    associated quantile estimate.

    Parameters
    ----------
    data : array_like
        data vector
    start : int
        lowest number of order statistics at which to plot a point
    end : int, optional
        highest number of order statistics at which to plot a point
    abline_y : float, optional
        value to be plotted as horizontal straight line
    ci : float
        probability for asymptotic confidence band
    ax : Axes, optional
        the Axes in which to plot the estimator
    """
    ordered = np.sort(data)[::-1]
    ordered = ordered[ordered > 0]
    n = len(ordered)
    k = np.arange(n)+1
    loggs = np.log(ordered)
    avesumlog = np.cumsum(loggs)/k
    xihat = np.hstack([np.nan, (avesumlog-loggs)[1:]])
    alphahat = 1/xihat
    y = alphahat
    ses = y/np.sqrt(k)
    if end is None:
        end = n-1
    x = np.arange(np.min([end, len(data)-1]), start, -1)
    y = y[x]
    qq = norm.ppf(1 - (1-ci)/2)
    u = y + ses[x] * qq
    l = y - ses[x] * qq
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, color='black', linewidth=1)
    ax.plot(x, u, color='red', linestyle='--', linewidth=1)
    ax.plot(x, l, color='red', linestyle='--', linewidth=1)
    if abline_y is not None:
        ax.axhline(abline_y, color='C0', linewidth=1)
    ax.set_ylabel('alpha (CI, p = '+str(ci)+")")
    ax.set_xlabel("Order Statistics")
    
    
def forecast_expert_ext(
    dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
     # Number of hours in a day 
    S = dat.shape[1]  # number of hours

    # Initialize forecast tensor with NaNs for each hour
    forecast = torch.full((S,), float('nan'), device=device)

    # Convert weekday dates to numeric values (1 = Monday, ..., 7 = Sunday)
    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    # Create weekday dummy variables for specified weekdays in `wd`
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)

    # Names of day-ahead forecast variables
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    
    # Names of fuel variables
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # Get column indices for fuels and DA variables
    reg_names = list(reg_names)
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    price_idx = reg_names.index("Price")
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    # Helper function to create 1D lagged tensor
    def get_lagged(Z, lag):
        if lag == 0:
           return Z
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])

    # Helper function to create 2D lagged tensor (for multivariate lags)
    def get_lagged_2d(Z, lag):
        if lag == 0:
           return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=Z.device), Z[:-lag]], dim=0)

    # Create lagged fuel variables for all specified lags and concatenate them
    mat_fuels = torch.cat(
        [get_lagged_2d(dat[:, 0, fuel_idx], lag=l) for l in fuel_lags], dim=1
    )

    # Lagged price from the last hour of the previous day
    price_last = get_lagged(dat[:, S - 1, price_idx], lag=1)

    # Container for coefficients
    num_features = len(wd) + len(price_s_lags) + len(fuel_names)*len(fuel_lags) + len(da_forecast_names)*len(da_lag) + 1
    coefs = torch.full((S, num_features), float('nan'), device=device)
     
     # Loop over each hour of the day to fit a separate regression model 
    for s in range(S):
         # Actual price (target variable) at hour s
        acty = dat[:, s, price_idx]

         # Lagged values of the current hour's price
        mat_price_lags = torch.stack([get_lagged(acty, lag) for lag in price_s_lags], dim=1)

        # Day-ahead forecast values at hour s
        mat_da_forecasts = dat[:, s, da_idx]
        
         # Create lags for each day-ahead forecast variable
        da_lagged_list = [
            torch.stack([get_lagged(mat_da_forecasts[:, i], lag) for lag in da_lag], dim=1)
            for i in range(len(da_forecast_names))
        ]
         # Combine all lagged day-ahead forecasts into one matrix
        da_all_var = torch.cat(da_lagged_list, dim=1)

        # Build the design matrix for regression
        if s == S - 1:
            # For last hour, exclude "Price last" predictor
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels], dim=1
                )
        else:
            # For all other hours, include "Price last"
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels, price_last.unsqueeze(1)], dim=1
                )

        # Filter out rows with missing data
        nan_mask = ~torch.any(torch.isnan(regmat), dim=1)
        regmat0 = regmat[nan_mask]



         # Standardize the data using mean and std of training part
        Xy = regmat0
        mean = Xy[:-1].mean(dim=0)
        std = Xy[:-1].std(dim=0)
        std[std == 0] = 1 # Prevent division by zero
        Xy_scaled = (Xy - mean) / std
          
        # Training data
        X = Xy_scaled[:-1, 1:].cpu().numpy()
        y = Xy_scaled[:-1, 0].cpu().numpy()
        x_pred = Xy_scaled[-1, 1:].cpu().numpy()
        
        # Fit linear regression model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        # Convert coefficients to tensor and clean NaNs
        coef = torch.tensor(model.coef_, dtype=torch.float32, device=device)
        coef[coef != coef] = 0  # Replace NaNs with 0

         # Compute the forecast (re-scale to original units)
        forecast[s] = (coef @ torch.tensor(x_pred, dtype=torch.float32, device=device)) * std[0] + mean[0]

        # Save coefficients
        if s == S - 1:
            coefs[s] = torch.cat([coef, torch.tensor([0.0], device=device)])
        else:
            coefs[s, :coef.numel()] = coef

    # Build coefficient dataframe
    regressor_names = (
        [f"Price lag {lag}" for lag in price_s_lags] +
        [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag] +
        [day_abbr[i - 1] for i in wd] +
        [f"{fuel} lag {lag}" for lag in fuel_lags for fuel in fuel_names] +
        ["Price last lag 1"]
    )
    # Convert coefficients to pandas DataFrame for inspection
    coefs_df = pd.DataFrame(coefs.cpu().numpy(), columns=regressor_names)
    
    # Return forecast and coefficients
    return {"forecasts": forecast, "coefficients": coefs_df}



def reg_matrix(dat_eval, days_eval, wd, fuel_lags, price_s_lags,
               da_lag, reg_names):
    
    
    # Number of series(hours)
    S = dat_eval.shape[1]

    # Prepare weekday dummy variables
    days_ext = days_eval
    # dt.weekday: Monday=0,... Sunday=6, add 1 to make Monday=1,... Sunday=7 
    weekdays_num = days_ext.dt.weekday  + 1 
    # Create dummy matrix WD: one column per weekday indicator
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])

    # Create column names for weekdays
    wd_columns = [f"WD_{x}" for x in wd]

    # Names of day-ahead forecast variables
    da_forecast_names =  ["Load_DA","Solar_DA", "WindOn_DA", "WindOff_DA"]

    # Names of fuel-related variables
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # Helper to compute lagged values of a 1D array Z by lag days
    def get_lagged(Z, lag):
        # Prepend 'lag' NaNs, then align the rest of data shifted
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) - lag)]))

    # Build fuel lag matrix: for each lag in fuel_lags, apply get_lagged to each fuel column
    mat_fuels = np.concatenate(
        [
            np.apply_along_axis(
                get_lagged, 0, dat_eval[:, 0, reg_names.isin(fuel_names)], lag=l
            )
            for l in fuel_lags
        ],
        axis=-1,
    )

    # Create column names for each fuel and lag combination
    fuel_columns = [f"{fuel}_lag_{l}" for l in fuel_lags for fuel in fuel_names]



    # Base matrix combining weekday dummies and fuel lags
    base_regmat = np.column_stack((WD, mat_fuels))
    column_base = wd_columns + fuel_columns 
    regmat1 = pd.DataFrame(base_regmat, columns=column_base)
    columns_base = regmat1.shape[1]


    # List to hold per-series DataFrames
    all_dataframes = []
    for s in range(S):

        # Extract actual price series for component s
        acty = dat_eval[:, s, reg_names == "Price"][..., 0]

        # Generate lagged prices for the price series
        mat_price_lags = np.transpose(
            [get_lagged(lag=lag, Z=acty) for lag in price_s_lags]
        )
        
        # Extract day-ahead forecast data for this series
        mat_da_forecasts = dat_eval[:, s, reg_names.isin(da_forecast_names)]
        
        
        # Create lagged day-ahead forecast variables up to lag 2
        stacked_da = []
        for i in range(len(da_forecast_names)):
            da_var = np.transpose([get_lagged(lag=lag, Z=mat_da_forecasts[:, i]) for lag in da_lag])
            stacked_da.append(da_var)  # Store the 2D arrays

        # Combine all lagged forecasts side by side
        da_all_var = np.hstack(stacked_da) 
        
        # Combine actual, price lags, and DA forecast lags into one matrix for series s
        regmat2 = np.column_stack((acty, mat_price_lags, da_all_var))


        # Build column names for this series: price and its lags, forecast lags
        columns = (
            [f"Price_s{s}"]
            + [f"Price_lag_{lag}_s{s}" for lag in price_s_lags]
            #+ [f"{name}_s{s}" for name in da_forecast_names]
            + [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag]
        )
        # Convert to DataFrame
        df = pd.DataFrame(regmat2, columns=columns)
        columns_s = df.shape[1]
        
        # Append to list
        all_dataframes.append(df)

    # Concatenate all DataFrames horizontally
    final_dataframe = pd.concat(all_dataframes, axis=1)

    # Merge series columns with base variables
    regmat = pd.concat([final_dataframe, regmat1], axis=1)
    columns_total = regmat.shape[1]
    
    # Return regression matrix and metadata
    return [regmat, columns_s, columns_base, columns_total, len(da_forecast_names)]



def expert_mlp_advhyper(
    train_loader,
    test_loader,
    num_feature,
    learning_rate,
    number_neurons,
    std_y,
    mean_y,
    weight_decay,
    beta1,
    beta2,
    num_epochs
):



    

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
           super(SimpleMLP, self).__init__()
        # hidden layer
           self.fc1 = nn.Linear(input_dim, hidden_dim)
        # activation for hidden layer
           self.act = nn.LeakyReLU()
        # output layer (no activation here)
           self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
        # apply first linear layer
          x = self.fc1(x)
        # apply activation only on hidden layer
          x= self.act(x)
        # apply output layer
          x = self.fc2(x)
          return x



    # input dimension
    input_dim = num_feature
    
    #middle dimension
    hidden_dim= number_neurons
    
    
    #output dimension
    output_dim = 24
    

    # Create the model
    model = SimpleMLP( input_dim, hidden_dim, output_dim).to(device)

    # Define the loss function for regression tasks            
    criterion = nn.MSELoss()

    # Configure the optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2),weight_decay=weight_decay)
    
    
    
    
    # Training loop
    # model.train()  # Set the model in training mode
    for epoch in range(num_epochs):
        model.train() 
        for X_train, y_train in train_loader:

            # Run a forward pass
            pred = model(X_train).squeeze(-1)
            # Compute loss and gradients
            loss = criterion(pred, y_train) 
            
            # Set the gradients to zero
            optimizer.zero_grad()
            # all gradiants are computed
            loss.backward()
            # Update the parameters
            optimizer.step()

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations during testing
        for X_test, y_test in test_loader:

            outputs = model(X_test).squeeze(-1)

            # Unstandardize outputs and y_test
            unstandardized_outputs = (outputs * std_y) + mean_y
            unstandardized_y_test = (y_test * std_y) + mean_y

            # calculate the squared error
            squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2


    
    return squared_errors, unstandardized_outputs,unstandardized_y_test
