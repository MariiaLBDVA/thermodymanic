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

# функция получает молярные массы компонентов, используемых  при расчете
def get_molar_masses():
    return {
        'Fe': Compound("Fe").molar_mass(),
        'S': Compound("S").molar_mass(),
        'As': Compound("As").molar_mass(),
        'K': Compound("K").molar_mass(),
        'NH4': Compound("NH4").molar_mass(),
        'H3O': 19.02,
        'FeS2': Compound("FeS2").molar_mass(),
        'FeAsS': Compound("FeAsS").molar_mass(),
        'Mg(OH)2': Compound("Mg(OH)2").molar_mass(),
        'H2SO4': Compound("H2SO4").molar_mass(),
        'MgSO4': Compound("MgSO4").molar_mass(),
    }

# основные параметры ярозитов
def get_jarosite_params():
    """Получить параметры ярозитов."""
    return {
        'K': {
            'name': 'K-ярозит',
            'dH0': 49683.33,  # Дж/моль
            'dS0': 233.52,    # Дж/(моль·К)
            'ion_name': 'K',
            'color': 'blue'
        },
        'H3O': {
            'name': 'H₃O-ярозит',
            'dH0': 90630,
            'dS0': 248.2166667,
            'ion_name': 'H3O',
            'color': 'red'
        },
        'NH4': {
            'name': 'NH₄-ярозит',
            'dH0': 58920,
            'dS0': 236.4866667,
            'ion_name': 'NH4',
            'color': 'purple'
        }
    } 


class CompositionCalculator:

    def __init__(self, molar_masses):
        self.M = molar_masses

    def calculate_ore_composition(self, Fe_w, S_w, As_w, K_w, NH4_w,
                                   mass_ore, Ж_Т, Mg_S, Fe_Ox,
                                   H2SO4_add_percent):
        
        mass_solid = mass_ore / (Ж_Т + 1)  # г
        mass_liquid = Ж_Т * mass_ore / 1000 * (Ж_Т + 1)  # кг

        # Массы основных компонентов заданному составу руды
        m_Fe = Fe_w * mass_solid / 100
        m_S = S_w * mass_solid / 100
        m_As = As_w * mass_solid / 100
        m_K = K_w * mass_solid / 100
        m_NH4 = NH4_w * mass_solid / 100
        
        m_H2SO4_add = H2SO4_add_percent * mass_liquid       
        m_MgOH_add = Mg_S * m_S

        # Расчёт арсенопирита
        n_FeAsS = m_As / self.M['As'] # количество арсенопирита равно количеству мышьяка
        m_Fe_FeAsS = n_FeAsS * self.M['Fe'] # количество железа в арсенопирите
        m_S_FeAsS = n_FeAsS * self.M['S'] # количество серы в арсенопирите
        
        m_Fe_left = m_Fe - m_Fe_FeAsS # масса железа вне арсенопирита
        m_S_left = m_S - m_S_FeAsS # масса серы вне арсенопирита
        
        # Расчёт пирита
        n_FeS2_from_Fe = m_Fe_left / self.M['Fe'] # количество пирита по железу
        n_FeS2_from_S = m_S_left / (2 * self.M['S']) # количество пирита по сере
        n_FeS2 = min(n_FeS2_from_Fe, n_FeS2_from_S) # количество пирита по железу
        
        n_S_excess = 0 # количество железа вне пирита и арсенопирита
        n_Fe_excess = 0 # количество серы вне пирита и арсенопирита
        
        # Определить избыток
        if n_FeS2_from_Fe < n_FeS2_from_S: # избыток серы
            n_S_excess = (m_S_left / self.M['S']) - (2 * n_FeS2)
            n_Fe_excess = 0
        else: # избыток железа
            n_S_excess = 0
            n_Fe_excess = (m_Fe_left / self.M['Fe']) - (n_FeS2)
     
        
        # сохраняем данные состава руды
        self.mass_solid = mass_solid
        self.mass_liquid = mass_liquid
        self.m_Fe = m_Fe
        self.m_S = m_S
        self.m_As = m_As
        self.m_K = m_K
        self.m_NH4 = m_NH4
        self.m_H2SO4_add = m_H2SO4_add
        self.m_MgOH_add = m_MgOH_add
        self.n_FeAsS = n_FeAsS
        self.n_FeS2 = n_FeS2
        self.n_S_excess = n_S_excess
        self.n_Fe_excess = n_Fe_excess

        return self

# класс определяет свойства ионов в растворе
class SolutionState:

    def ion_params(self):
        ion_names = [
            "H",
            "SO4",
            "HSO4",
            "Fe2",
            "Fe3",
            "K",
            "Na",
            "NH4",
            "H3O",
            "Mg",
            "MgSO4"
        ]
        charges = {  
            "H": 1,
            "SO4": -2,
            "HSO4": -1,
            "Fe2": 2,
            "Fe3": 3,
            "K": 1,
            "Na": 1,
            "NH4": 1,
            "H3O": 1,
            "Mg": 2,
            "MgSO4": 0  
        }
        
        ion_radii = {
            "H": 4.78e-8,
            "SO4": 5.31e-8,
            "HSO4": 4.5e-8,
            "Fe2": 5.08e-8,
            "Fe3": 9.0e-8,
            "K": 3.71e-8, 
            "Na": 4.32e-8,
            "NH4": 2.5e-8,
            "H3O": 0,
            "Mg": 8.0e-8,
            "MgSO4": 3.0e-8
    }
        ions = {
            name: {
                "z": charges[name], 
                "C": 0,
                "gamma": 1,
                "a": 0,
                "r": ion_radii[name]
            }
            for name in ion_names
        }
        
        return ions


