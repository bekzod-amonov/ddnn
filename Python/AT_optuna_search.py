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
import locale
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List

import optuna
from optuna.trial import TrialState

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, StudentT
from scipy.stats import johnsonsu 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distribution = "Normal" # <- change accordingly
paramcount = {"Normal": 2,
              "StudentT": 3,
              "JSU": 4,
              "SinhArcsinh": 4,
              "NormalInverseGaussian":4,
              "Point": None}

retrain_no = 3
INP_SIZE    = 221
# for optuna search
activations  = ['sigmoid', 'relu', 'elu', 'leaky_relu', 'tanh', 'softplus']
param_names = ['loc', 'scale', 'tailweight', 'skewness']
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
val_window = val_days // retrain_no

repo_root = Path(__file__).resolve().parent.parent
trial_dir = repo_root / "trialfiles"
trial_dir.mkdir(parents=True, exist_ok=True)

bzn = "AT" 
data = pd.read_csv( f"{repo_root}/Datasets/{bzn}.csv", index_col=0)
#time_utc = pd.to_datetime(data[]) <- part of DST_trafo
data.index = [datetime.strptime(e, "%Y-%m-%d %H:%M:%S") for e in data.index]
#data = DST_trafo(X=data[:, 1:],Xtime=) <- apply DST later
        
df_train_val = data.iloc[data.index.searchsorted(
  INIT_DATE_EXP):data.index.searchsorted(INIT_DATE_EXP)+train_val_days*24] # take the first 4 years for now;

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
      tailweight = 1 + 3 * F.softplus(self.self_tailweight(h2))
      skewness = self.head_skewness(h2)
      params = {'loc':loc,'scale':scale, 'tailweight':tailweight, 'skewness':skewness}
      return (params, (h1,h2)) if self.return_hidden else params
    
    if self.distribution == "NormalInverseGaussian":
      loc   = self.head_loc(h2)
      scale = 1e-3 + 3 * F.softplus(self.head_scale(h2)) 
      tailweight = 1 + 3 * F.softplus(self.tailweight(h2))
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
        
def objective(trial):
  
  print(f"==================== Step 1 - obtaining hyperparameters ====================")
  
  Y_train_val = np.zeros((train_val_days,24))
  Y_val = np.zeros((val_window, 24))
  
  for d in range(Y_train_val.shape[0]):    
    Y_train_val[d, :] = df_train_val.loc[df_train_val.index[d*24:(d+1)*24],'Price'].to_numpy()
  # Y_train_t = Y_train_t[7:, :] # skip the first 7 days due to lagged features
  for d in range(Y_val.shape[0]):
    Y_val[d, :] = df_train_val.loc[df_train_val.index[(d+train_days)*24:(d+1+train_days)*24], 'Price'].to_numpy()
  
  X_train_val = np.zeros((train_val_days,INP_SIZE))
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
  if trial.suggest_categorical('price_D-1', binopt):
    colmask[:24]   = [True] * 24
  if trial.suggest_categorical('price_D-2', binopt):
    colmask[24:48] = [True] * 24
  if trial.suggest_categorical('price_D-3', binopt):
    colmask[48:72] = [True] * 24
  if trial.suggest_categorical('price_D-7', binopt):
    colmask[72:96] = [True] * 24
  if trial.suggest_categorical('load_DA', binopt):
    colmask[96:120] = [True] * 24
  if trial.suggest_categorical('load_DA_D-1', binopt):
    colmask[120:144] = [True] * 24
  if trial.suggest_categorical('load_DA_D-7', binopt):
    colmask[144:168] = [True] * 24
  if trial.suggest_categorical('RES_DA_D', binopt):
    colmask[168:192] = [True] * 24
  if trial.suggest_categorical('RES_DA_D-1', binopt):
    colmask[192:216] = [True] * 24
  if trial.suggest_categorical('EUA', binopt):
    colmask[216] = True
  if trial.suggest_categorical('Coal', binopt):
    colmask[217] = True
  if trial.suggest_categorical('NGas', binopt):
    colmask[218] = True
  if trial.suggest_categorical('Oil', binopt):
    colmask[219] = True
  if trial.suggest_categorical('Week_Day_Dummy', binopt):
    colmask[220] = True
  X_train_val = X_train_val[:, colmask]
  
  X_train_val_all = X_train_val.copy()
  Y_train_val_all = Y_train_val.copy()
  metrics_sub = []
  
  # widhts/lr/ac
  widths = (
    trial.suggest_int("neurons_1", 64, 1024),
    trial.suggest_int("neurons_2", 64, 1024))
  
  activation_function = (
    trial.suggest_categorical('activation_1', activations),
    trial.suggest_categorical('activation_2', activations))
  
  use_batchnorm = True
  return_hidden = True
  output_dim    = 24
  
  use_dropout = trial.suggest_categorical('dropout', binopt)
  dropout_p = (0.0 if not use_dropout else trial.suggest_float('dropout_p', 0.0, 0.5))
  
  regularize_h1_activation = trial.suggest_categorical('regularize_h1_activation', binopt)
  h1_activation_rate = (0.0 if not regularize_h1_activation else 
                        trial.suggest_float('h1_activation_rate_l1',  1e-5, 1e-2, log=True))
  
  regularize_h1_kernel = trial.suggest_categorical('regularize_h1_kernel', binopt)
  h1_kernel_rate = (0.0 if not regularize_h1_kernel else 
                    trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e-2, log=True))
  
  regularize_h2_activation = trial.suggest_categorical('regularize_h2_activation', binopt)
  h2_activation_rate = (0.0 if not regularize_h2_activation else 
                        trial.suggest_float('h2_activation_rate_l1',  1e-5, 1e-2, log=True))
  
  regularize_h2_kernel = trial.suggest_categorical('regularize_h2_kernel', binopt)
  h2_kernel_rate = (0.0 if not regularize_h2_kernel else 
                    trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e-2, log=True))
  
  head_l1_rates = {}
  param_names = ['loc', 'scale', 'tailweight', 'skewness', 'df']
  if paramcount[distribution] is not None:
    for p in range(paramcount[distribution]):
      regularize_param_kernel = trial.suggest_categorical(f'regularize_{param_names[p]}', binopt)
      param_kernel_rate = (0.0 if not regularize_param_kernel else
                          trial.suggest_float(f'{param_names[p]}_rate_l1', 1e-5, 1e-2, log=True))
      head_l1_rates[param_names[p]] = param_kernel_rate
      
  learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
  #weight_decay  = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True) do not use it 
  epochs = trial.suggest_int('epochs', 30, 80) # set some random range
  
  
  for train_no in range(retrain_no):
    
    # cross-validation split
    start       = val_window * train_no
    X_train_val = X_train_val_all[start:train_days+start, :]
    X_val       = X_train_val_all[train_days+start:train_days+start+val_window, :]
    Y_train_val = Y_train_val_all[start:train_days+start, :]
    Y_val       = Y_train_val_all[train_days+start:train_days+start+val_window, :] 
    X_train_val = X_train_val[7:train_days, :]
    Y_train_val = Y_train_val[7:train_days, :]
    
    # build dataloader
    X_train_t = torch.as_tensor(X_train_val, dtype=torch.float32)
    Y_train_t = torch.as_tensor(Y_train_val, dtype=torch.float32)
    X_val_t   = torch.as_tensor(X_val, dtype=torch.float32) 
    Y_val_t   = torch.as_tensor(Y_val, dtype=torch.float32) 
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=32, shuffle=False)
    
    # build model
    model = ProbMLP(
               input_dim     = X_train_t.shape[1],
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
        pred_train, (h1, h2) = model(x) 
        
        if distribution == 'Point':
          pred = pred_train # tensor(B,24)
          loss_train = mae(pred, y) # MAE
        elif distribution == 'Normal':
          params = pred_train # dict
          dist   = Normal(loc=params["loc"], scale=params["scale"]) # change to: dist = model.make_dist(params)
          loss_train   = (-dist.log_prob(y)).mean() # NLL
        elif distribution == 'StudentT':
          params = pred_train
          dist   = StudentT(df=params["df"], loc=params["loc"], scale=params["scale"]) # change to: dist = model.make_dist(params)
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
    with torch.no_grad():
      X_val_t = X_val_t.to(device)
      Y_val_t = Y_val_t.to(device)
      pred_val, (h1v, h2v) = model(X_val_t)
      
      if distribution == 'Point':
        loss_val = mae(pred_val, Y_val_t).item()
      elif distribution == 'Normal':
        dist     = Normal(loc=pred_val['loc'], scale=pred_val['scale'])
        loss_val = (-dist.log_prob(Y_val_t)).mean().item()
      elif distribution == 'StudentT':
        dist     = StudentT(df=pred_val["df"], loc=pred_val["loc"], scale=pred_val["scale"]) # change to: dist = model.make_dist(params)
        loss_val = (-dist.log_prob(Y_val_t)).mean().item() # negative log likelihood
      metrics_sub.append(loss_val)
  return float(np.mean(metrics_sub))

study_name = f'FINAL_{bzn}_selection_prob_{distribution.lower()}'
db_path = (repo_root / "trialfiles" / f"{study_name}.db").resolve()
storage = f"sqlite:///{db_path.as_posix()}"
study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=10, show_progress_bar=False)
best_params = study.best_params
print(best_params)
print(study.trials_dataframe())
      
out = Path(f"{repo_root}/trialfiles") / f"{study_name}_best_params.json"
out.write_text(json.dumps(best_params, indent=2))
print("Saved:", out)      
    
      
      
    
      
    
    
    
      
      
    
               
               
            
   
   
   