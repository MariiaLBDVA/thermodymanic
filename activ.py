 
from scipy import constants
import numpy as np
from sympy import inverse_cosine_transform
from main_file import (
    SolutionState
)
from scipy.optimize import least_squares
# расчет активностей ионов 

class ActivityCalculator:
    # константы в сгс
    k_CGS = 1.38054e-16
    e_CGS = 4.80320425e-10
    
    # берем значения для воды из интерполяционных функций
    def __init__(self, density, epsilon):
        self.get_density = density
        self.get_epsilon = epsilon
    
    # Возвращаем радиус иона из словаря параметров
    def get_ion_radius(self, ion_params):
        return ion_params.get("r", 0)
    
    # расчет параметра А для заданной температуры и параметров воды, (кг^1/2 * моль ^ 1/2)
    def calculate_A(self, T, density, epsilon):
        numerator = (2 * np.pi * constants.N_A)**0.5 * (self.e_CGS**3) * (density**0.5)
        denominator = 2.302585 * 1000**0.5 * (epsilon * self.k_CGS * T)**1.5
        return numerator / denominator
    
    # расчет параметра В для заданной температуры и параметров воды, (кг^1/2 * моль ^ 1/2 / см)
    def calculate_B(self, T, density, epsilon):
        numerator = 8 * np.pi * constants.N_A * density * self.e_CGS**2
        denominator = epsilon * self.k_CGS * T * 1000
        return (numerator / denominator)**0.5

    # расчет ионной силы раствора для всех ионов в растворе
    def calculate_ionic_strength(self, ions):
        I = 0
        for params in ions.items():
            I += 0.5 * params["C"] * params["z"]**2
        return I
    
    # уравнение Дебая-Хюккеля
    def debye_huckel_term(self, z, I, A, B, a):
        denominator = 1 + a * B * np.sqrt(I)
        return -A * z**2 * np.sqrt(I) / denominator
    
    # объединяем все функции и выполняем расчет логарифма коэффициента активности для данной температуры и набора ионов
    def calculate(self, T, ions):
        density = self.get_density(T)
        epsilon = self.get_epsilon(T)

        A = self.calculate_A(T, density, epsilon)
        B = self.calculate_B(T, density, epsilon)
        I = self.calculate_ionic_strength(ions)

        for params in ions.items():
            z = params["z"] # заряд иона из словаря
            a = self.get_ion_radius(params) # ионный радиус из словаря
            dh_term = self.debye_huckel_term(z, I, A, B, a)
            si_term = -0.4 * params["SO4"]["C"]

            log_gamma = dh_term + si_term
            gamma = 10**log_gamma

            # обновляем словарь ионов
            params["gamma"] = gamma
            params["a"] = params["C"] * gamma  # активность * концентрация

        # возвращаем обновленный словарь
        return ions, I


def update_activities(state, T, activity_model):

    # state.concentrations — словарь ионов с их текущими концентрациями
    ions_dict = state.concentrations

    # вызываем расчёт активности
    ions_updated, I = activity_model.calculate(T, ions_dict, C_SO4=ions_dict.get("SO4", 0))

    # обновляем состояния
    state.gamma = {ion: params["gamma"] for ion, params in ions_updated.items()}
    state.I = I

    state.activities = {ion: params["C"] * params["gamma"] for ion, params in ions_updated.items()}

    return state


class SpeciationSolver:

    def __init__(self, T, state, activity_model, h2so4_interp=None, mgso4_interp=None, add_Mg=False, max_iter=50, tol=1e-8):
        
        self.T = T
        self.state = state
        self.activity_model = activity_model
        self.h2so4_interp = h2so4_interp
        self.mgso4_interp = mgso4_interp
        self.add_Mg = add_Mg
        self.max_iter = max_iter
        self.tol = tol

        # история ионов и ионной силы
        self.I_history = []
        self.gamma_history = []

        # сохраняем исходные суммы для масс-баланса - общее количество всех ионов постоянно
        self.total_H = state.concentrations.get("H", 0) + state.concentrations.get("HSO4", 0)
        self.total_SO4 = state.concentrations.get("SO4", 0) + state.concentrations.get("HSO4", 0) + state.concentrations.get("MgSO4", 0)
        self.total_Mg = state.concentrations.get("Mg", 0) + state.concentrations.get("MgSO4", 0)

    def calculate(self):
        # решаем уравнения до достижения максимального количетсва итераций или максимальной точности
        for iteration in range(self.max_iter):

            # обновляем gamma и активности
            self.state = update_activities(self.state, self.T, self.activity_model)

            self.I_history.append(self.state.I)
            self.gamma_history.append(self.state.gamma.copy())

            # решаем систему равновесий
            if not self.add_Mg:
                new_conc = self._solve_system_no_mg()
            else:
                if self.total_H < 1e-5:
                    new_conc = self._solve_system_only_mg()
                else:
                    new_conc = self._solve_system_full()

            # проверка сходимости
            deltas = [abs(self.state.concentrations[k] - new_conc.get(k, 0)) for k in new_conc]
            if max(deltas) < self.tol:
                self.state.concentrations.update(new_conc)
                break

            # обновляем концентрации для следующей итерации
            self.state.concentrations.update(new_conc)

            # проверка констант диссоциации и массы
            self._check_dissociation_constants()
            self._check_mass_balance()

        return self.state

    # уравнения 
    def _solve_system_no_mg(self):
        H = self.state.concentrations["H"]
        SO4 = self.state.concentrations["SO4"]
        HSO4 = self.state.concentrations.get("HSO4", 0)
        γH = self.state.gamma["H"]
        γSO4 = self.state.gamma["SO4"]
        γHSO4 = self.state.gamma["HSO4"]
        
        K_HSO4 = self.h2so4_interp.get_K(self.T, self.state.I)

        def eq(vars):
            SO4_val, H_val, HSO4_val = vars
            return [
                SO4_val + HSO4_val - self.total_SO4,
                (SO4_val * γSO4) * (H_val * γH) - K_HSO4 * HSO4_val*γHSO4
            ]
        x0 = [SO4, H, HSO4]

        res = least_squares(eq, x0, bounds=(0, np.inf))
        SO4_new, H_new, HSO4_new = res.x
        
        return {
            "SO4": SO4_new,
            "H": H_new,
            "HSO4": HSO4_new
        }

    def _solve_system_only_mg(self):
        H = self.state.concentrations.get("H", 0)
        SO4 = self.state.concentrations.get("SO4", 0)
        Mg = self.state.concentrations.get("Mg", 0)
        MgSO4 = self.state.concentrations.get("MgSO4", 0)
        HSO4 = self.state.concentrations.get("HSO4", 0)
        
        γSO4 = self.state.gamma.get("SO4", 1.0)
        γMg = self.state.gamma.get("Mg", 1.0)
        γH = self.state.gamma.get("H", 1.0)
        γHSO4 = self.state.gamma.get("HSO4", 1.0)
        
        K_MgSO4 = self.mgso4_interp.get_K(self.T)
        K_HSO4 = self.h2so4_interp.get_K(self.T, self.state.I)
        
        def eq(vars):
            SO4_val, H_val, HSO4_val, Mg_val, MgSO4_val = vars
            return [
                (SO4_val * γSO4) * (H_val * γH) - K_HSO4 * HSO4_val*γHSO4,
                SO4_val + MgSO4_val + HSO4_val - self.total_SO4,
                Mg_val + MgSO4_val - self.total_Mg,
                (Mg_val * γMg) * (SO4_val * γSO4) - K_MgSO4 * MgSO4_val
            ]

        x0 = [SO4,  H, HSO4, Mg, MgSO4]
        res = least_squares(eq, x0, bounds=(0, np.inf))
        SO4_new, H_new, HSO4_new, Mg_new, MgSO4_new = res.x

        return {
            "SO4": SO4_new,
            "H": H_new,
            "HSO4": HSO4_new,
            "Mg": Mg_new,
            "MgSO4": MgSO4_new,
        }


    def _check_dissociation_constants(self):
        if self.state.activities.get('HSO4', 0) > 1e-15:
            K_calc = (self.state.activities.get('SO4', 0) *
                      self.state.activities.get('H', 0) /
                      self.state.activities.get('HSO4', 1))
            print(f"K(HSO4-) = {K_calc:.6f}")
        if self.add_Mg and self.state.activities.get('MgSO4', 0) > 1e-15:
            K_calc = (self.state.activities.get('Mg2', 0) *
                      self.state.activities.get('SO4', 0) /
                      self.state.activities.get('MgSO4', 1))
            print(f"K(MgSO4) = {K_calc:.6f}")

    def _check_mass_balance(self):
        final_H_total = self.state.concentrations.get('H', 0) + self.state.concentrations.get('HSO4', 0)
        final_SO4_total = (self.state.concentrations.get('SO4', 0) +
                           self.state.concentrations.get('HSO4', 0) +
                           self.state.concentrations.get('MgSO4', 0))
        if abs(final_H_total - self.total_H) > 1e-6:
            print("⚠️ Баланс по H нарушен!")
        if abs(final_SO4_total - self.total_SO4) > 1e-6:
            print("⚠️ Баланс по SO4 нарушен!")