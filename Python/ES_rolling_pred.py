import pandas as pd
import numpy as np

from multiprocessing import Pool

from datetime import datetime, timedelta, date
import logging
import sys
import os, getpass
import glob
import time
import math
import json
from multiprocessing import freeze_support
import locale
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List

import optuna
from optuna.trial import TrialState
import optuna

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, StudentT
from scipy.stats import johnsonsu 

import os
from multiprocessing import Pool
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distribution = "Normal" # <- change accordingly
paramcount = {"Normal": 2,
              "StudentT": 3,
              "JSU": 4,
              "SinhArcsinh": 4,
              "NormalInverseGaussian":4,
              "Point": None}

retrain_no = 13
INP_SIZE   = 221
bzn        = "AT" 
# for optuna search
activations  = ['sigmoid', 'relu', 'elu', 'leaky_relu', 'tanh', 'softplus']
# for internal mapper in pytorch
_ACTS = {'sigmoid':    nn.Sigmoid,
         'relu':       nn.ReLU,
         'elu':        nn.ELU,
         'leaky_relu': nn.LeakyReLU,
         'tanh':       nn.Tanh,
         'softplus':   nn.Softplus,
         'softmax':    nn.Softmax}
 
binopt      = [True, False]

START_YEAR      = 2020   
VAL_START_YEAR  = 2022   
TRAIN_END_YEAR  = 2023   
FINAL_END_YEAR  = 2025   

INIT_DATE_EXP   = pd.Timestamp(f'{START_YEAR}-01-01 00:00:00')     # hyper-parameter tuning: training start date
VAL_INIT_DATE   = pd.Timestamp(f'{VAL_START_YEAR}-12-28 00:00:00') # hyper-parameter tuning: training end date
TRAIN_END_DATE  = pd.Timestamp(f'{TRAIN_END_YEAR}-12-27 00:00:00') # hyper-parameter tuning: validation end date
FINAL_DATE_EXP  = pd.Timestamp(f"{FINAL_END_YEAR}-12-31 00:00:00") # rolling window: evaluation end date

train_val_days = (TRAIN_END_DATE - INIT_DATE_EXP).days
train_days  = (VAL_INIT_DATE - INIT_DATE_EXP).days
val_days   = (TRAIN_END_DATE - VAL_INIT_DATE).days
#val_window = val_days // retrain_no

repo_root = Path(__file__).resolve().parent.parent
trial_dir = repo_root / "trialfiles"
trial_dir.mkdir(parents=True, exist_ok=True)

if not os.path.exists(f'{repo_root}/forecasts_probNN_{distribution.lower()}'):
    os.mkdir(f'{repo_root}/forecasts_probNN_{distribution.lower()}')

if not os.path.exists(f'{repo_root}/distparams_probNN_{distribution.lower()}'):
    os.mkdir(f'{repo_root}/distparams_probNN_{distribution.lower()}')

data = pd.read_csv( f"{repo_root}/Datasets/{bzn}_all.csv", index_col=0)
data.index = [datetime.strptime(e, "%Y-%m-%d %H:%M:%S") for e in data.index]
data = data.sort_index()

if distribution not in paramcount: raise ValueError('Incorrect distribution defined')
 
class ProbMLP(nn.Module):
  
  def __init__(self,
               input_dim: int,
               widths: tuple[int, int], #(neurons_1, neurons_2)
               activations: tuple[str, str],
               output_dim: int,
               use_dropout: bool,
               dropout_p: float,
               distribution: str,
               return_hidden: bool=True,
               use_batchnorm: bool=True):
    
    super().__init__()
    
    w1, w2 = widths
    a1, a2 = activations
    
    self.distribution = distribution
    self.return_hidden = return_hidden
    
    self.bn = nn.BatchNorm1d(input_dim) if use_batchnorm else None
    self.dropout = nn.Dropout(dropout_p) if use_dropout else None
    
    self.fc1 = nn.Linear(input_dim, w1)
    self.act1 = _ACTS[a1.lower()]()
    
    self.fc2 = nn.Linear(w1, w2)
    self.act2 = _ACTS[a2.lower()]()
    
    if distribution == 'Point':
      self.head = nn.Linear(w2, 24)
    
    elif distribution == 'Normal':
      self.head_loc   = nn.Linear(w2, 24)
      self.head_scale = nn.Linear(w2, 24)
    
    elif distribution == 'StudentT':
      self.head_loc   = nn.Linear(w2, 24)
      self.head_scale = nn.Linear(w2, 24)
      self.head_df    = nn.Linear(w2, 24)
    
    # apply from scipy library      
    elif distribution == 'JSU':
      self.head_loc   = nn.Linear(w2, 24)
      self.head_scale = nn.Linear(w2, 24)
      self._head_tailweight = nn.Linear(w2, 24)
      self.head_skewness   = nn.Linear(w2, 24)
      
    # apply from ___ library      
    elif distribution == 'SinhArcsinh':
      self.head_loc   = nn.Linear(w2, 24)
      self.head_scale = nn.Linear(w2, 24)
      self.head_tailweight = nn.Linear(w2, 24)
      self.head_skewness   = nn.Linear(w2, 24)
      
    # apply from ____ library      
    elif distribution == 'NormalInverseGaussian':
      self.head_loc   = nn.Linear(w2, 24)
      self.head_scale = nn.Linear(w2, 24)
      self.head_tailweight = nn.Linear(w2, 24)
      self.head_skewness   = nn.Linear(w2, 24)
        
    else: 
      raise ValueError(f'unsupported distribution: {distribution}.')
    
  def forward(self, x):
    
    if self.bn is not None:
      x = self.bn(x)
    
    if self.dropout is not None:
      x = self.dropout(x)
    
    # hidden layers h1,h2
    h1 = self.act1(self.fc1(x))
    if self.dropout is not None:
      h1 = self.dropout(h1)
      
    h2 = self.act2(self.fc2(h1))
    if self.dropout is not None:
      h2 = self.dropout(h2)
    
    # point forecast head
    if self.distribution == 'Point':
      y = self.head(h2)
      return (y, (h1, h2)) if self.return_hidden else y
    
    # distribution heads
    if self.distribution == "Normal":
      loc    = self.head_loc(h2) 
      scale  = 1e-3 + 3 * F.softplus(self.head_scale(h2))
      params = {'loc':loc,'scale':scale}
      return (params, (h1,h2)) if self.return_hidden else params
    
    if self.distribution == "StudentT":
      loc   = self.head_loc(h2)
      scale = 1e-3 + 3 * F.softplus(self.head_scale(h2)) 
      df    = 1 + 3 * F.softplus(self.head_df(h2))
      params = {'loc':loc,'scale':scale,'df':df} 
      return (params, (h1,h2)) if self.return_hidden else params
    
    if self.distribution == "JSU":
      loc   = self.head_loc(h2)
      scale = 1e-3 + 3 * F.softplus(self.head_scale(h2)) 
      tailweight = 1 + 3 * F.softplus(self.head_tailweight(h2))
      skewness = self.head_skewness(h2)
      params = {'loc':loc,'scale':scale, 'tailweight':tailweight, 'skewness':skewness}
      return (params, (h1,h2)) if self.return_hidden else params
    
    if self.distribution == "SinhArcsinh":
      loc   = self.head_loc(h2)
      scale = 1e-3 + 3 * F.softplus(self.head_scale(h2)) 
      tailweight = 1 + 3 * F.softplus(self.head_tailweight(h2))
      skewness = self.head_skewness(h2)
      params = {'loc':loc,'scale':scale, 'tailweight':tailweight, 'skewness':skewness}
      return (params, (h1,h2)) if self.return_hidden else params
    
    if self.distribution == "NormalInverseGaussian":
      loc   = self.head_loc(h2)
      scale = 1e-3 + 3 * F.softplus(self.head_scale(h2)) 
      tailweight = 1 + 3 * F.softplus(self.head_tailweight(h2))
      skewness = self.head_skewness(h2)
      params = {'loc':loc,'scale':scale, 'tailweight':tailweight, 'skewness':skewness}
      return (params, (h1,h2)) if self.return_hidden else params
    
  def make_dist(self, params):
        # params -> torch.distributions object
        if self.distribution == "Normal":
            return Normal(loc=params["loc"], scale=params["scale"])
        if self.distribution == "StudentT":
            return StudentT(df=params["df"], loc=params["loc"], scale=params["scale"])
        # integrate other distribution families manually or from other libraries!
        if self.distribution == "JSU":
            return None
        if self.distribution == "SinhArcsinh":
            return None
        if self.distribution == "NormalInverseGaussian":
            return None
        
def rolling_window(input):
  
  best_params, day_no = input   
  start = data.index.searchsorted(INIT_DATE_EXP) + day_no * 24
  df_train_val = data.iloc[start : start + train_val_days*24 + 24]
  Y_train_val  = np.zeros((train_val_days,24))
  
  train_start = df_train_val.index[0]
  train_end   = df_train_val.index[train_val_days*24 - 1]          
  pred_day    = df_train_val.index[train_val_days*24]              
  total_rolls = (end - base)//24 - train_val_days
  if day_no % 25 == 0 or day_no == total_rolls - 1: 
    print(f"[{day_no+1}/{total_rolls}] train: {train_start:%Y-%m-%d} -> {train_end:%Y-%m-%d %H:%M} | predict day: {pred_day:%Y-%m-%d}")

  
  for d in range(Y_train_val.shape[0]):    
    Y_train_val[d, :] = df_train_val.loc[df_train_val.index[d*24:(d+1)*24],'Price'].to_numpy()
  Y_train_val = Y_train_val[7:, :] # skip the first 7 days due to lagged features
  
  X_train_val = np.zeros((train_val_days+1,INP_SIZE))
  for d in range(7, X_train_val.shape[0]):
    X_train_val[d, :24] = df_train_val.loc[df_train_val.index[(d-1)*24:(d*24)],'Price'].to_numpy()     # D-1 price
    X_train_val[d, 24:48] = df_train_val.loc[df_train_val.index[(d-2)*24:((d-1)*24)],'Price'].to_numpy() # D-2 price
    X_train_val[d, 48:72] = df_train_val.loc[df_train_val.index[((d-3)*24):((d-2)*24)],'Price'].to_numpy()   # D-3 price
    X_train_val[d, 72:96] = df_train_val.loc[df_train_val.index[((d-7)*24):((d-6)*24)],'Price'].to_numpy() # D-7 price
    X_train_val[d, 96:120] = df_train_val.loc[df_train_val.index[(d*24):((d+1)*24)],'Load_DA'].to_numpy()        # D Load_DA
    X_train_val[d, 120:144] = df_train_val.loc[df_train_val.index[((d-1)*24):(d*24)],'Load_DA'].to_numpy()     # D-1 Load_DA
    X_train_val[d, 144:168] = df_train_val.loc[df_train_val.index[((d-7)*24):((d-6)*24)],'Load_DA'].to_numpy() # D-7 Load_DA
    X_train_val[d, 168:192] = df_train_val.loc[df_train_val.index[(d*24):((d+1)*24)],'Renewables_DA_Forecast'].to_numpy() # D Renewables summ DA
    X_train_val[d, 192:216] = df_train_val.loc[df_train_val.index[((d-1)*24):(d*24)],'Renewables_DA_Forecast'].to_numpy() # D-1 Renewables summ DA
    X_train_val[d, 216] = df_train_val.loc[df_train_val.index[(d-2)*24:(d-1)*24:24],'EUA'].to_numpy().item()  # D-2 EUA
    X_train_val[d, 217] = df_train_val.loc[df_train_val.index[(d-2)*24:(d-1)*24:24],'Coal'].to_numpy().item() # D-2 Coal
    X_train_val[d, 218] = df_train_val.loc[df_train_val.index[(d-2)*24:(d-1)*24:24],'NGas'].to_numpy().item() # D-2 NGas
    X_train_val[d, 219] = df_train_val.loc[df_train_val.index[(d-2)*24:(d-1)*24:24],'Oil'].to_numpy().item()  # D-2 Oil
    X_train_val[d, 220] = df_train_val.index[d].weekday()
  
  # include feature selection into hyper-parameter space  
  colmask = [False] * INP_SIZE
  if best_params['price_D-1']: colmask[:24]   = [True] * 24
  if best_params['price_D-2']: colmask[24:48] = [True] * 24
  if best_params['price_D-3']: colmask[48:72] = [True] * 24
  if best_params['price_D-7']: colmask[72:96] = [True] * 24
  if best_params['load_DA']  : colmask[96:120] = [True] * 24
  if best_params['load_DA_D-1']: colmask[120:144] = [True] * 24
  if best_params['load_DA_D-7']: colmask[144:168] = [True] * 24
  if best_params['RES_DA_D']: colmask[168:192] = [True] * 24
  if best_params['RES_DA_D-1']:colmask[192:216] = [True] * 24
  if best_params['EUA']: colmask[216] = True
  if best_params['Coal']: colmask[217] = True
  if best_params['NGas']: colmask[218] = True
  if best_params['Oil']:  colmask[219] = True
  if best_params['Week_Day_Dummy']: colmask[220] = True
  X_train_val = X_train_val[:, colmask]
  
  X_predict = X_train_val[-1:, :]
  X_train_val = X_train_val[7:-1, :]
  
  # widhts/lr/ac
  widths = (best_params["neurons_1"],best_params["neurons_2"])
  activation_function = (best_params['activation_1'],best_params['activation_2'])
  
  use_batchnorm = True
  return_hidden = True
  output_dim    = 24
  
  use_dropout = best_params['dropout']
  dropout_p = best_params['dropout_p'] if use_dropout else 0.0
  
  regularize_h1_activation = best_params['regularize_h1_activation']
  h1_activation_rate = (0.0 if not regularize_h1_activation else best_params['h1_activation_rate_l1'])
  
  regularize_h1_kernel = best_params['regularize_h1_kernel']
  h1_kernel_rate = (0.0 if not regularize_h1_kernel else best_params['h1_kernel_rate_l1'])
  
  regularize_h2_activation = best_params['regularize_h2_activation']
  h2_activation_rate = (0.0 if not regularize_h2_activation else best_params['h2_activation_rate_l1'])
  
  regularize_h2_kernel = best_params['regularize_h2_kernel']
  h2_kernel_rate = (0.0 if not regularize_h2_kernel else best_params['h2_kernel_rate_l1'])
  
  head_l1_rates = {}
  param_names = ['loc', 'scale', 'tailweight', 'skewness', 'df']
  if paramcount[distribution] is not None:
    for p in range(paramcount[distribution]):
      regularize_param_kernel = best_params[f'regularize_{param_names[p]}']
      param_kernel_rate = (0.0 if not regularize_param_kernel else best_params[f'{param_names[p]}_rate_l1'])
      head_l1_rates[param_names[p]] = param_kernel_rate
      
  learning_rate = best_params['learning_rate']
  #weight_decay  = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True) do not use it 
  epochs = best_params['epochs']
  
  
  VAL_DATA = 0.2  
  N = X_train_val.shape[0] 
  perm = np.random.permutation(N)
  cut = int((1.0 - VAL_DATA) * N)
  train_idx = perm[:cut]
  val_idx   = perm[cut:]
  X_t = torch.as_tensor(X_train_val, dtype=torch.float32)
  Y_t = torch.as_tensor(Y_train_val, dtype=torch.float32)
  full_ds = TensorDataset(X_t, Y_t)
  train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
  val_ds   = torch.utils.data.Subset(full_ds, val_idx.tolist())
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
  val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
  
  # build model
  model = ProbMLP(
              input_dim     = X_t.shape[1],
              widths        = widths,
              activations   = activation_function,
              output_dim    = output_dim,
              use_batchnorm = use_batchnorm,
              use_dropout   = use_dropout,
              dropout_p     = dropout_p,
              return_hidden = return_hidden,
              distribution = distribution,).to(device)
  
  # I need to pass the following stuff into for _ in range(epochs) and I do not know how to do it.
  optim   = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
  mae = nn.L1Loss()
  model.to(device)
  
  for _ in range(epochs):
    model.train()
    for x, y in train_loader:
      x = x.to(device)
      y = y.to(device)
      optim.zero_grad()
      params_pred, (h1, h2) = model(x)   
    
      if distribution == 'Point':
        loss_train = mae(params_pred, y) 
      elif distribution == 'Normal':
        dist   = Normal(loc=params_pred["loc"], scale=params_pred["scale"]) # change to: dist = model.make_dist(params)
        loss_train   = (-dist.log_prob(y)).mean() # NLL
      elif distribution == 'StudentT':
        dist   = StudentT(df=params_pred["df"], loc=params_pred["loc"], scale=params_pred["scale"]) # change to: dist = model.make_dist(params)
        loss_train   = (-dist.log_prob(y)).mean() # NLL

      # hidden layer weight (kernel) penalty
      weight_l1 = torch.tensor(0.0, device=device)
      if regularize_h1_kernel:
        weight_l1 = weight_l1 + h1_kernel_rate * model.fc1.weight.abs().sum()
      if regularize_h2_kernel:
        weight_l1 = weight_l1 + h2_kernel_rate * model.fc2.weight.abs().sum()
      
      # activation function penalty
      act_l1 = torch.tensor(0.0, device=device)
      if regularize_h1_activation:
        act_l1 = act_l1 + h1_activation_rate * h1.abs().sum()
      if regularize_h2_activation:
        act_l1 = act_l1 + h2_activation_rate * h2.abs().sum()
        
      # head l1 penalty
      head_l1 = torch.tensor(0.0, device=device)
      head_modules = {
        'loc': model.head_loc,
        'scale': model.head_scale,
        'df': getattr(model, 'head_df', None),
        'tailweight': getattr(model, 'head_tailweight', None),
        'skewness': getattr(model, 'head_skewness', None)}
      
      for name, rate in head_l1_rates.items():
        m = head_modules.get(name, None)
        if (m is not None) and (rate > 0.0):
          head_l1 = head_l1 + rate * m.weight.abs().sum()
        
      loss_reg = loss_train + weight_l1 + act_l1 + head_l1
      loss_reg.backward()
      optim.step()
      
  model.eval()
  #val_losses = []
  with torch.no_grad():

    Xf_t = torch.as_tensor(X_predict, dtype=torch.float32).to(device) # (1, p)
    params_pred, _ = model(Xf_t) # dict of (1,24) tensors
    if distribution == "Normal":
        getters = {"loc": params_pred["loc"], "scale": params_pred["scale"]}
    elif distribution == "StudentT":
        getters = {"loc": params_pred["loc"], "scale": params_pred["scale"], "df": params_pred["df"]}
    else:
        raise ValueError("Implement getters for this distribution")
      
    params_out = {k: [float(e) for e in v.detach().cpu().numpy()[0]] for k, v in getters.items()}
    day_key = datetime.strftime(df_train_val.index[-24], "%Y-%m-%d")
    json.dump(params_out,open(os.path.join(f"{repo_root}/distparams_probNN_{distribution.lower()}", day_key), "w"))
    
    if distribution == "Normal":
        dist = Normal(loc=getters["loc"], scale=getters["scale"])
    elif distribution == "StudentT":
        dist = StudentT(df=getters["df"], loc=getters["loc"], scale=getters["scale"])

    # 10000 MC scenarios; result shape (10000, 1, 24) -> squeezed to (10000, 24) -> (24,) in pred_mean
    pred = dist.sample((10000,)).squeeze(1).detach().cpu().numpy()
    pred_mean = pred.mean(axis=0)
    np.savetxt(os.path.join(f"{repo_root}/forecasts_probNN_{distribution.lower()}", day_key),pred,delimiter=",",fmt="%.3f")
    predDF = pd.DataFrame(index=df_train_val.index[-24:])
    predDF["real"] = df_train_val.loc[df_train_val.index[-24:], "Price"].to_numpy()
    predDF["forecast"] = pred_mean
        
  return predDF
            
study_name = f'FINAL_{bzn}_selection_prob_{distribution.lower()}'
db_path = (repo_root / "trialfiles" / f"{study_name}.db").resolve()
storage_name = f"sqlite:///{db_path.as_posix()}"
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
print(study.trials_dataframe())
best_params = study.best_params
print(best_params)
# best_params = json.loads((repo_root / "trialfiles" / f"{study_name}_best_params.json").read_text())

base = data.index.searchsorted(INIT_DATE_EXP)
end  = data.index.searchsorted(FINAL_DATE_EXP)
total_days = (end - base) // 24
inputlist = [(best_params, day) for day in range(total_days - train_val_days)]
print(len(inputlist))

workers = max(os.cpu_count() // 4, 1)
if __name__ == "__main__":
    print("==================== Step 2 - NN evaluation ====================")
    with Pool(max(os.cpu_count() // 4, 1)) as p:
        _ = p.map(rolling_window, inputlist)
        
        