 
class ActivityCalculator:
    ion_radii = {
        'H': 4.78e-8, 'K': 3.71e-8, 'Na': 4.32e-8,
        'Fe3': 9.0e-8, 'Fe2': 5.08e-8, 'SO4': 5.31e-8,
        'HSO4': 4.5e-8, 'NH4': 2.5e-8, 'Mg2': 8.0e-8,
        'MgSO4': 3.0e-8, 'OH': 4.0e-8
    }
    k_CGS = 1.38054e-16
    e_CGS = 4.80320425e-10
    
    def __init__(self, water_density_func, water_dielectric_func):
        self.get_density = water_density_func
        self.get_epsilon = water_dielectric_func
    
    def get_ion_radius(self, ion_name):
        return self.ion_radii.get(ion_name, 4.0e-8)
    
    def calculate_A(self, T, density, epsilon):
        numerator = (2 * np.pi * constants.N_A)**0.5 * (self.e_CGS**3) * (density**0.5)
        denominator = 2.302585 * 1000**0.5 * (epsilon * self.k_CGS * T)**1.5
        return numerator / denominator
    
    def calculate_B(self, T, density, epsilon):
        numerator = 8 * np.pi * constants.N_A * density * self.e_CGS**2
        denominator = epsilon * self.k_CGS * T * 1000
        return (numerator / denominator)**0.5
    
    def calculate_ionic_strength(self, concentrations, charges):
        return 0.5 * sum(c * z**2 for c, z in zip(concentrations, charges))
    
    def debye_huckel_term(self, z, I, A, B, a):
        sqrt_I = np.sqrt(I)
        denominator = 1 + a * B * sqrt_I
        return -A * z**2 * sqrt_I / denominator
    
    def calculate(self, T, concentrations, charges, ion_names, C_SO4=0):
        density = self.get_density(T)
        epsilon = self.get_epsilon(T)
        
        A = self.calculate_A(T, density, epsilon)
        B = self.calculate_B(T, density, epsilon)
        I = self.calculate_ionic_strength(concentrations, charges)
        
        gammas = []
        dh_terms = []
        si_terms = []
        
        for ion_name, z in zip(ion_names, charges):
            a = self.get_ion_radius(ion_name)
            dh_term = self.debye_huckel_term(z, I, A, B, a)
            si_term = -0.4 * C_SO4
            
            log_gamma = dh_term + si_term
            gamma = 10**log_gamma
            
            gammas.append(gamma)
            dh_terms.append(dh_term)
            si_terms.append(si_term)
        
        return {
            'gamma': gammas,
            'I': I,
            'A': A,
            'B': B,
            'dh_terms': dh_terms,
            'si_terms': si_terms
        }

class SpeciationCalculator:

    def __init__(self, T, state, add_Mg=False,
                 h2so4_interp=None, mgso4_interp=None):
        self.T = T
        self.state = state
        self.h2so4_interp = h2so4_interp
        self.mgso4_interp = mgso4_interp
        

def update_activities(state, T, activity_model):

    concentrations = state.get_concentration_array()

    result = activity_model.calculate(
            T,
            concentrations,
            state.charges,
            state.ion_names
        )

    gamma = result["gamma"]

    state.gamma = dict(zip(state.ion_names, gamma))

    state.I = result["I"]

    state.activities = {
        ion: state.concentrations[ion] * state.gamma[ion]
        for ion in state.ion_names
    }

    return state

class SpeciationSolver:

    def __init__(self, T, state, h2so4_interp, mgso4_interp):

        self.T = T
        self.state = state

        self.h2so4_interp = h2so4_interp
        self.mgso4_interp = mgso4_interp

    def _solve_system_no_mg(self):
        H = self.state.concentrations["H"]
        SO4 = self.state.concentrations["SO4"]

        γH = self.state.gamma["H"]
        γSO4 = self.state.gamma["SO4"]

        K = self.h2so4_interp.get_K(self.T, self.state.I)

        def equation(x):

            HSO4 = x

            H_new = H - HSO4
            SO4_new = SO4 - HSO4

            return (H_new * γH) * (SO4_new * γSO4) - K * HSO4

        sol = least_squares(equation, SO4 / 2, bounds=(0, SO4))

        HSO4 = sol.x[0]

        new_concentrations = {

            "H": H - HSO4,
            "SO4": SO4 - HSO4,  
            "HSO4": HSO4
        }

        return new_concentrations
    
    def _solve_system_only_mg(self):
        H = self.state.concentrations["H"]
        SO4 = self.state.concentrations["SO4"]
        Mg = self.state.concentrations["Mg"]
        MgSO4 = self.state.concentrations.get("MgSO4", 0)

        γH = self.state.gamma["H"]
        γSO4 = self.state.gamma["SO4"]
        γMg = self.state.gamma["Mg"]

        K_H2SO4 = self.h2so4_interp.get_K(self.T, self.state.I)
        K_MgSO4 = self.mgso4_interp.get_K(self.T)

        def eq(vars):

            SO4_val, Mg_val, MgSO4_val = vars
            return [
                SO4_val + MgSO4_val - self.total_SO4,
                Mg_val + MgSO4_val - self.total_Mg,
                (Mg_val*γMg)*(SO4_val*γSO4) - K_MgSO4 * MgSO4_val
            ]
            
        x0 = [SO4, Mg, MgSO4]
        res = least_squares(eq, x0, bounds=(0, np.inf))
        SO4_new, Mg_new, MgSO4_new = res.x
        
        self.speciation["H"] = 0
        self.speciation["HSO4"] = 0
        self.speciation["SO4"] = SO4_new
        self.speciation["Mg2"] = Mg_new
        self.speciation["MgSO4"] = MgSO4_new

        new_concentrations = {

            "H": H - HSO4,
            "SO4": SO4 - HSO4,  
            "HSO4": HSO4
        }
        
        return new_concentrations
 
    
    def _check_dissociation_constants(self):
        """Проверить константы диссоциации по активностям"""
        if self.activities.get('HSO4', 0) > 1e-15:
            K_calc = (self.activities.get('SO4', 0) * 
                     self.activities.get('H', 0) / 
                     self.activities.get('HSO4', 1))
            self.dissociation['K_HSO4'] = K_calc
            print(f"  K(HSO₄⁻) = {K_calc:.6f}")
        
        if self.add_Mg and self.activities.get('MgSO4', 0) > 1e-15:
            K_calc = (self.activities.get('Mg2', 0) * 
                     self.activities.get('SO4', 0) / 
                     self.activities.get('MgSO4', 1))
            self.dissociation['K_MgSO4'] = K_calc
            print(f"  K(MgSO₄) = {K_calc:.6f}")
    
    def _check_mass_balance(self):
        """Проверить соблюдение материального баланса"""
        final_H_total = self.speciation.get('H', 0) + self.speciation.get('HSO4', 0)
        final_SO4_total = (self.speciation.get('SO4', 0) + 
                          self.speciation.get('HSO4', 0) + 
                          self.speciation.get('MgSO4', 0))
        
        if abs(final_H_total - self.total_H) > 1e-6:
            print(f"\n⚠️  ВНИМАНИЕ: Баланс по водороду нарушен!")
        if abs(final_SO4_total - self.total_SO4) > 1e-6:
            print(f"\n⚠️  ВНИМАНИЕ: Баланс по сере нарушен!")
    
    def calculate(self):
        
        for iteration in range(self.max_iter):
    
            active_ions, active_conc, active_charges = self._get_active_components()
            
            act_result = self._update_gamma(active_ions, active_conc, active_charges)
            
            K_HSO4, K_MgSO4 = self._get_equilibrium_constants()
            
            H, SO4, HSO4, Mg, MgSO4 = self._get_current_concentrations()
            
            γH = self.gamma.get("H", 1.0)
            γSO4 = self.gamma.get("SO4", 1.0)
            γHSO4 = self.gamma.get("HSO4", 1.0)
            γMg = self.gamma.get("Mg2", 1.0)
            
            try:
                if not self.add_Mg:
                    H_new, SO4_new, HSO4_new = self._solve_system_no_mg(
                        H, SO4, HSO4, γH, γSO4, γHSO4, K_HSO4
                    )
                    new_vals = [H_new, SO4_new, HSO4_new]
                    old_vals = [H, SO4, HSO4]
                    names = ['H', 'SO4', 'HSO4']
                    
                else:
                    if self.total_H < 1e-5:
                        SO4_new, Mg_new, MgSO4_new = self._solve_system_only_mg(
                            SO4, Mg, MgSO4, γMg, γSO4, K_MgSO4
                        )
                        new_vals = [SO4_new, Mg_new, MgSO4_new]
                        old_vals = [SO4, Mg, MgSO4]
                        names = ['SO4', 'Mg', 'MgSO4']
                    else:
                        new_vals = self._solve_system_full(
                            H, SO4, HSO4, Mg, MgSO4,
                            γH, γSO4, γHSO4, γMg, K_HSO4, K_MgSO4
                        )
                        old_vals = [H, SO4, HSO4, Mg, MgSO4]
                        names = ['H', 'SO4', 'HSO4', 'Mg', 'MgSO4']
                        
            except Exception as e:
                print(f"⚠️  Ошибка в итерации {iteration}: {e}")
                break
            
            self.I_history.append(self.I)
            
            # 7. Проверка сходимости
            deltas = self._calculate_deltas(old_vals, new_vals, names)
            
            if max(deltas) < self.tol:
                print(f"  ✅ Сходимость за {iteration+1} итераций")
                break
            
            # 8. Обновление активностей и проверка констант
            self._update_activities()
            self._check_dissociation_constants()
            self._check_mass_balance()
        

