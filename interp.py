from unittest import result
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from chemlib import Compound
from scipy import constants
from scipy.optimize import least_squares


def clean_experimental_data_local_outliers(temp, concentration, solubility, 
                                           z_thresh=3, k=15):
    X = np.column_stack([temp, concentration])
    
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X)
    
    mask_good = np.ones(len(solubility), dtype=bool)
    
    for i in range(len(solubility)):
        neigh_idx = idx[i, 1:]  # исключаем саму точку
        local_mean = np.mean(solubility[neigh_idx])
        local_std = np.std(solubility[neigh_idx])
        
        if local_std > 0:
            z = np.abs(solubility[i] - local_mean) / local_std
            if z > z_thresh:
                mask_good[i] = False
    
    return (
        temp[mask_good],
        concentration[mask_good],
        solubility[mask_good]
    )



class WaterPropertiesInterpolator:
    """Интерполятор для плотности и диэлектрической проницаемости воды."""
    def __init__(self):
        """Инициализация с табличными данными при 10 МПа."""
        self.table_temps_K = np.array([298.15, 328.15, 373.15, 513.15])
        self.table_epsilon = np.array([88.3, 69.67, 55.4, 30.79])
        self.table_density = np.array([0.72068, 0.65672, 0.55203, 0.27483])
        
        self.epsilon_interp = interp1d(
            self.table_temps_K, self.table_epsilon,
            kind='quadratic', fill_value='extrapolate' 
        )
        self.density_interp = interp1d(
            self.table_temps_K, self.table_density,
            kind='linear', fill_value='extrapolate'
        )
    
    def get_density(self, T_K):
        return float(self.density_interp(T_K))
    
    def get_dielectric(self, T_K):
        return float(self.epsilon_interp(T_K))

class Interpolator:
    
    def __init__(self, file_path, sheet_name):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.scaler = None
        self.rbf_interpolator = None
    
    def prepare_data(self):
        raise NotImplementedError

class MgSO4ConstantInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values + 273.15
        logK = self.df.iloc[:, 1].values
        K = 10 ** logK
        
        data_df = pd.DataFrame({'temp': temp, 'K': K})
        avg_df = data_df.groupby('temp', as_index=False).mean()
        
        self.temp = avg_df['temp'].values
        self.K = avg_df['K'].values
        
        self.interp_func = interp1d(
            self.temp, self.K,
            kind='quadratic', fill_value='extrapolate'
        )
    
    def get_K(self, T_K):
    
        return float(self.interp_func(T_K))
    
class MgSO4SolubilityInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values
        MgSO4_sol = self.df.iloc[:, 1].values
        H2SO4_conc = self.df.iloc[:, 2].values
    
        self.points = np.column_stack([temp, H2SO4_conc])
        self.MgSO4_sol = MgSO4_sol
        
        self.scaler = StandardScaler()
        self.points_normalized = self.scaler.fit_transform(self.points)
        
        self.rbf_interpolator = RBFInterpolator(
            self.points_normalized, MgSO4_sol,
            kernel='linear', smoothing=0.1
        )
    
    def get_sol(self, T_K, H2SO4_conc):
        point = np.array([[T_K, H2SO4_conc]])
        point_normalized = self.scaler.transform(point)
        return float(self.rbf_interpolator(point_normalized)[0])

class H2SO4ConstantInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values + 273.15
        logK = self.df.iloc[:, 1].values
        ionic_strength = self.df.iloc[:, 2].values
        
        K = 10 ** logK
        
        self.points = np.column_stack([temp, ionic_strength])
        self.K = K
        
        self.scaler = StandardScaler()
        self.points_normalized = self.scaler.fit_transform(self.points)
        
        self.rbf_interpolator = RBFInterpolator(
            self.points_normalized, K,
            kernel='cubic', smoothing=0.1
        )
    
    def get_K(self, T_K, ionic_strength):
        point = np.array([[T_K, ionic_strength]])
        point_normalized = self.scaler.transform(point)
        return float(self.rbf_interpolator(point_normalized)[0])
