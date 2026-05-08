"""
Calculates entanglement asymmetry of of deformed toric code (Castelnovo Chamon 2008)
Runs DMRG (twice) with TeNPy to get gnd state of deformed toric code Hamiltonian H(beta) 
Spatial manifold is torus of size Lx x Ly, gnd state breaks 1-form Wilson loop symmetry
From gnd state contracts MPS to get rho_A on cylinder subregion of size width x Ly
Entanglement Asymmetry = S(rho_A_symmetrized) - S(rho_A)
    [ Requires directory 'data/' to save data, best if run on HPC cluster ]
    [ On 8 core node: Lx = 20 ran in max 5-6 hrs, used max 10-12 GB of memory ]
""" 
# general imports
import sys
import numpy as np
from itertools import combinations, accumulate
from timeit import default_timer as timer
import pickle
# tenpy imports
import tenpy as tp
MPS = tp.networks.mps.MPS
dmrg = tp.algorithms.dmrg
ToricCode = tp.models.toric_code.ToricCode
TermList = tp.networks.terms.TermList
# Show DMRG log (comment line below to hide)
tp.tools.misc.setup_logging(to_stdout="INFO") 

# Set deformation parameter beta
# Use regular toric code (beta = 0) by default
beta_vals = np.linspace(0.0, 0.93, 15)
betaind = 0
beta = beta_vals[betaind]

# Set system, subsystem sizes
Lx, Ly = 8, 4
width = 2
print(f'Lx = {Lx}, Ly = {Ly}, width = {width}, beta = {beta}')
starting_site = Ly + 2*Ly
sites_wilson = [starting_site + k * (2 * Ly) for k in range(width)]
sites_A  = np.arange(starting_site, sites_wilson[-1] + Ly , 1)
print(f"Number of sites in cylinder subregion A: {len(sites_A)}")

# schedule of max bond dimension per sweep
chi_maxes = [16, 32, 64, 100, 128, 160, 180, 200, 220, 256]
num_sweeps =[8,   8, 14,  14,  14,  12,  10,  10,  10,  12]
starts = [0] + list(accumulate(num_sweeps[:-1]))
chi_list = dict(zip(starts, chi_maxes))
print("Sweep number : bond dimension", "\n", chi_list)

# DMRG params and sweep schedule
dmrg_params = {'mixer': True, 
    'chi_list': chi_list,
    'trunc_params': {'chi_max': 256, 'svd_min': 1.e-8},
    'max_E_err': 1.e-8, 'max_S_err':1.e-6, 
    'min_sweeps':80, 'max_sweeps':120}

"""Order of lattice sites in MPS (both direction PBC) 
|         |         |         
1         5         9        
|         |         |         
. -- 3 -- . -- 7 -- . -- 11 -- 
|         |         |         
0         4         8        
|         |         |         
. -- 2 -- . -- 6 -- . -- 10 --       

"""

# Toric code with 'external fields' that are wilson and 't Hooft loops
# swapped X and Z compared to our asymmetry paper conventions
class ExtendedToricCode(ToricCode):

    def init_terms(self, model_params):
        super().init_terms(model_params)

        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        J_Wy = model_params.get('J_Wy', 0.)
        J_Ty = model_params.get('J_Ty', 0.) 
        J_Wx = model_params.get('J_Wx', 0.)

        # Wilson loop in y-direction (Z string on lattice along y direction at fixed x)
        x0, u0 = 0, 0  # vertical links
        self.add_local_term(-J_Wy, [('Sigmaz', [x0, y, u0]) for y in range(Ly)])

        # Wilson loop in x-direction (Z string on lattice along x direction at fixed y)
        y0, u0 = 0, 1  # horizontal links
        self.add_local_term(-J_Wx, [('Sigmaz', [x, y0, u0]) for x in range(Lx)])

        # 't Hooft loop (X string on dual lattice along y direction at fixed x)
        x0, u0 = 0, 1  # horizontal links
        self.add_local_term(-J_Ty, [('Sigmax', [x0, y, u0]) for y in range(Ly)])

    def wilson_loop_y(self, psi):
        Ly = self.lat.shape[1]
        x, u = 0, 0
        Wy = TermList.from_lattice_locations(
            self.lat,
            [[("Sigmaz", [x, y, u]) for y in range(Ly)]]
        )
        return psi.expectation_value_terms_sum(Wy)[0].item()

    def hooft_loop_y(self, psi):
        Ly = self.lat.shape[1]
        x, u = 0, 1
        Ty = TermList.from_lattice_locations(
            self.lat,
            [[("Sigmax", [x, y, u]) for y in range(Ly)]]
        )
        return psi.expectation_value_terms_sum(Ty)[0].item()

    def wilson_loop_x(self, psi):
        Lx = self.lat.shape[0]
        x, u = 0, 1  # horizontal links at y=0
        Wx = TermList.from_lattice_locations(
            self.lat,
            [[("Sigmaz", [x, 0, u]) for x in range(Lx)]]
        )
        return psi.expectation_value_terms_sum(Wx)[0].item()


# Deformed toric code using (Huxford, Nguyen, Kim 2023) conventions
class DeformedToricCode(ExtendedToricCode):

    def init_terms(self, model_params):
        super().init_terms(model_params)
        
        # One new parameter: additional deformation term
        # beta_critical ~ 0.44
        beta = model_params.get('beta', 0.)
        cb = np.cosh(beta)
        sb = np.sinh(beta)

        # add term \sum_{vertices} exp[-beta * (Z_1 + Z_2 + Z_3 + Z_4)]
        # expand into sums of products of Z around vertices
        # (cb * Id_1+ sb * Z_1) * (cb * Id_2 + sb * Z_2) * ...
        star = [
            ('Sigmaz', [0,  0], 1),
            ('Sigmaz', [0,  0], 0),
            ('Sigmaz', [-1, 0], 1),
            ('Sigmaz', [0, -1], 0),
        ]
        for r in range(4 + 1):
            coeff = ((-sb)**r) * (cb**(4-r))
            for subset in combinations(range(4), r):
                ops = [star[k] for k in subset]
                if ops:
                    # products of a single Z are treated separately 
                    if len(ops) == 1:
                        op, lat_idx, u = ops[0]
                        # Cheap trick: add Id on a different site to make it a two-site coupling
                        dummy = ('Id', [lat_idx[0], lat_idx[1]], 1 - u)
                        self.add_multi_coupling(coeff, [ops[0], dummy])
                    # products of multiple Z's are simple
                    else:
                        self.add_multi_coupling(coeff, ops)

# given psi and y-cycle cylinder A of size (width x Ly)
# outputs S(rho_A_sym) under Wilson loop symmetry operator W_x
def symmetrized_entropy(psi, Lx, Ly, width):
    starting_site = Ly + 2*Ly
    sites_wilson = [starting_site + k * (2 * Ly) for k in range(width)] # qubits in Wilson loop restricted to A
    sites_A  = np.arange(starting_site, sites_wilson[-1] + Ly , 1)    
    n_sites = len(sites_A)
    dim_A = 2 ** n_sites
    print("\nStarting symmetrizing")

    # Get rho_A via MPS contraction 
    rho_npc = psi.get_rho_segment(sites_A)
    # Force leg order: p0, p1, ..., p{k}, p0*, p1*, ..., p{k}*
    ket_labels = [f'p{i}' for i in range(n_sites)]
    bra_labels = [f'p{i}*' for i in range(n_sites)]
    rho_npc.itranspose(ket_labels + bra_labels)
    rho_A = rho_npc.to_ndarray().reshape(2**n_sites, 2**n_sites)
    print("got rhoA")
    
    # Construct W_{x,A} as matrix via tensor product
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I = np.eye(2)
    WA = np.array([[1.]])
    for k in range(n_sites):
        op = Z if sites_A[k] in sites_wilson else I
        WA = np.kron(WA, op)
    print("got W")
    
    # symmetrize using wilson loop restricted to region A
    rho_A_sym = 0.5 * (rho_A + WA @ rho_A @ WA)
    print("symmetrized")
    
    # get von Neumann entropy
    evals = np.linalg.eigvalsh(rho_A_sym)
    print("diagonalized")
    evals = evals[evals > 1e-14]
    S_sym = -np.dot(evals, np.log(evals))
    return S_sym

# checks that gnd state maximally breaks the global symmetry before calculating asymmetry in region
def asymmetry_deformed_TC(dmrg_params, Lx, Ly, width=2, beta=0.):
    # parameters
    model_params = {'Lx': Lx, 'Ly': Ly, 'bc_MPS': 'finite', 'bc_x': 'periodic', 'bc_y': 'periodic',
                    'conserve': None, 'J_Wy': 0, 'J_Ty': 0, 'J_Wx': 0, 'beta': beta}
    
    # subregion A, y-cycle cylinder, starts at index 3*Ly and goes <width> sites to the right
    # don't include vertical bonds on edges of A inside A (cheaper)
    starting_site = Ly + 2*Ly
    sites_wilson = [starting_site + k * (2 * Ly) for k in range(width)]
    sites_A  = np.arange(starting_site, sites_wilson[-1] + Ly , 1)
    print(f"Number of sites in A: {len(sites_A)}")

    # RUN DMRG WITH TWO WILSON LOOPS IN HAM TO GET GLOBAL SECTOR RIGHT
    # Run DMRG twice to choose sector 1/sqrt(2) * [(Wy = 1, Wx = 1) + (Wy = 1, Wx = -1)]
    print("Running DMRG twice to get right sector...")
    
    # DMRG 1: ++ sector (J_Wy=5, J_Wx=5) 
    print("Running ++ DMRG")
    model_params['J_Wy'] = 5.
    model_params['J_Wx'] = 5.
    H_pp = DeformedToricCode(model_params)
    product_state = [0] * H_pp.lat.N_sites
    psi_pp = MPS.from_product_state(H_pp.lat.mps_sites(), product_state.copy(), bc=H_pp.lat.bc_MPS)
    dmrg.run(psi_pp, H_pp, dmrg_params)
    print("Finished ++ DMRG")
    print("psi.chi: ", psi_pp.chi)
    Wy = H_pp.wilson_loop_y(psi_pp)
    Wx = H_pp.wilson_loop_x(psi_pp)
    print(f"After ++ DMRG: <W_y> = {Wy}, <W_x> = {Wx}")

    # save MPS 
    with open(f'data/psi_pp_beta{betaind}.p', 'wb') as h:
        pickle.dump(psi_pp, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("DUMPED state pp")

    # DMRG 2: +- sector (J_Wy=5, J_Wx=-5)
    print("\nRunning +- DMRG")
    model_params['J_Wy'] = 5.
    model_params['J_Wx'] = -5.
    H_pm = DeformedToricCode(model_params)
    psi_pm = MPS.from_product_state(H_pm.lat.mps_sites(), product_state.copy(), bc=H_pm.lat.bc_MPS)
    dmrg.run(psi_pm, H_pm, dmrg_params)
    print("psi.chi: ", psi_pm.chi)
    
    # save MPS 
    with open(f'data/psi_pm_beta{betaind}.p', 'wb') as h:
        pickle.dump(psi_pm, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("DUMPED state pm")

    Wy = H_pm.wilson_loop_y(psi_pm)
    Wx = H_pm.wilson_loop_x(psi_pm)
    print(f"After +- DMRG: <W_y> = {Wy}, <W_x> = {Wx}")

    # SSB gnd state is superposition: psi = 1/sqrt(2) * (psi_pp + psi_mp)
    # TeNPy: sum two MPS then SVD-compress back to chi_max
    # IMPORTANT: Make sure that the energies of psi_pp and psi_mp are the same!
    psi = psi_pp.add(psi_pm, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2))
    info = psi.compress_svd({'chi_max': 256})
    print("Superposition state formed and compressed")   
    print(f"Truncated singular value weight: {info.eps} ")
    print(f"Overlap pre/post truncation: {info.ov} ")
    print("Bond dimensions: ", psi.chi)

    # save MPS 
    with open(f'data/psi_beta{betaind}.p', 'wb') as h:
        pickle.dump(psi, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("DUMPED state final")

    # Check Wilson and 't Hooft loops after preparing superposition
    model_params['J_Wx'] = 0.
    model_params['J_Wy'] = 0.
    H0 = DeformedToricCode(model_params)
    Wy = H0.wilson_loop_y(psi)
    Ty = H0.hooft_loop_y(psi)
    Wx = H0.wilson_loop_x(psi)
    print(f"After superposition: <W_y> = {Wy}, <W_x> = {Wx}, <T_y> = {Ty}")

    # Compute entanglement entropy (regular)
    segment = np.array(sites_A) - sites_A[0]
    S = psi.entanglement_entropy_segment(segment=segment, first_site=[sites_A[0]])[0]
    print("\nS =", S)
    print("S(ordinary toric code) ~", (2 * Ly - 2) * np.log(2))

    # compute rhoA_sym and symmetrized entanglement entropy 
    S_sym = symmetrized_entropy(psi, Lx, Ly, width)
    Asym = S_sym - S
    print(f"\nS_symmetrized = {S_sym}")
    print("Delta S_A = ", Asym)
    print("Reference (log 2):", np.log(2))

    return {'Wy': Wy, 'Wx': Wx, 'Ty': Ty, 
            'S': S, 'S_sym': S_sym, 'Asym': Asym}

# generate data for one beta value
def gen_data():
    start = timer()
    print(f"Starting beta = {beta}\n\n")
    
    result = asymmetry_deformed_TC(dmrg_params, Lx=Lx, Ly=Ly, width=width, beta=beta)
    asym = result['Asym']
    print(f"beta = {beta}, Delta S_A = {asym}")

    with open(f'data/asymdata_Lx{Lx}_Ly{Ly}_width{width}_betaind{betaind}.p', 'wb') as h:
        pickle.dump(result, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("DUMPED")
    
    end = timer()
    print("\nTIME ELAPSED: ", end-start, "\n")

def main():
    gen_data()

if __name__ == "__main__":
    main()
