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
def get_jarosite_params(molar_masses):
    """Получить параметры ярозитов."""
    return {
        'K': {
            'name': 'K-ярозит',
            'dH0': 49683.33,  # Дж/моль
            'dS0': 233.52,    # Дж/(моль·К)
            'M_cation': molar_masses['K'],
            'ion_name': 'K',
            'charge': 1,
            'color': 'blue'
        },
        'H3O': {
            'name': 'H₃O-ярозит',
            'dH0': 90630,
            'dS0': 248.2166667,
            'M_cation': molar_masses['H3O'],
            'ion_name': 'H3O',
            'charge': 1,
            'color': 'red'
        },
        'NH4': {
            'name': 'NH₄-ярозит',
            'dH0': 58920,
            'dS0': 236.4866667,
            'M_cation': molar_masses['NH4'],
            'ion_name': 'NH4',
            'charge': 1,
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

        # Массы основных компонентов
        m_Fe = Fe_w * mass_solid / 100
        m_S = S_w * mass_solid / 100
        m_As = As_w * mass_solid / 100
        m_K = K_w * mass_solid / 100
        m_NH4 = NH4_w * mass_solid / 100
        
        m_H2SO4_add = H2SO4_add_percent * mass_liquid       
        m_MgOH_add = Mg_S * m_S

        # Расчёт арсенопирита
        n_FeAsS = m_As / self.M['As']
        m_Fe_FeAsS = n_FeAsS * self.M['Fe']
        m_S_FeAsS = n_FeAsS * self.M['S']
        m_Fe_left = m_Fe - m_Fe_FeAsS
        m_S_left = m_S - m_S_FeAsS
        
        # Расчёт пирита
        n_FeS2_from_Fe = m_Fe_left / self.M['Fe']
        n_FeS2_from_S = m_S_left / (2 * self.M['S'])
        n_FeS2 = min(n_FeS2_from_Fe, n_FeS2_from_S)
        
        n_S_excess = 0
        n_Fe_excess = 0
        # Определить избыток
        if n_FeS2_from_Fe < n_FeS2_from_S:
            n_S_excess = (m_S_left / self.M['S']) - (2 * n_FeS2)
            n_Fe_excess = 0
        else:
            n_S_excess = 0
            n_Fe_excess = (m_Fe_left / self.M['Fe']) - (n_FeS2)
    
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

class SolutionState:

    def __init__(self, ion_names, concentrations, charges):

        self.ion_names = ion_names
        self.charges = charges

        self.concentrations = dict(zip(ion_names, concentrations))

        self.gamma = {}
        self.activities = {}

        self.I = None

    def get_concentration_array(self):

        return [
            self.concentrations[i]
            for i in self.ion_names
        ]
