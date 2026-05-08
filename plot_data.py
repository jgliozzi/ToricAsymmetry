# Plot data produced by deformed_asym.py
# plot shows asymmetry of SSB ground state vs beta for different Lx
# In low beta (topological phase), asym ~ log 2 indep of Lx
# In high beta (trivial phase), asym --> 0 as Lx --> \infty
import numpy as np
from matplotlib import pyplot as plt
import pickle
plt.rcParams.update({
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18
})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['text.usetex'] = True

# Data parameters
beta_vals = np.linspace(0.0, 0.93, 15)
Ly = 4
width = 2
	
# Create Plot
fig, ax = plt.subplots(figsize=(7,5))

supdata = []
Lx_vals = np.arange(4, 22, 4) # Lx = [4, 8, 12, 16, 20]
for Lx in Lx_vals:
	print(f'Lx = {Lx}')
	data = []
	for beta_ind, beta in enumerate(beta_vals):
		
		try:
			with open(f'data/asymdata_Lx{Lx}_Ly{Ly}_width{width}_betaind{beta_ind}.p', 'rb') as h:
				res = pickle.load(h)
			Wy = res['Wy']
			Wx = res['Wx']
			Ty = res['Ty']
			S = res['S']
			S_sym = res['S_sym'] 
			Asym = res['Asym']
			data.append([Wy, Wx, Ty, S, S_sym, Asym])

		except FileNotFoundError:
			print("FAILURE: beta index", beta_ind)
			data.append(6*[np.nan]) 

	print("Done importing data")
	supdata.append(data)

	wilson_y, wilson_x, hooft_y, S_vals, Ssym_vals, asyms = zip(*data)
	ax.plot(beta_vals, asyms, 'o-', 
			linewidth=3, markersize=7, label=fr'$L_x = $ {Lx}')

ax.set_xlabel(r'$\beta$', fontsize=24)
ax.set_ylabel(r'$\Delta S_A$', fontsize=24, color='black')
ax.set_ylim([0, 0.75])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.legend(fontsize=16)

# Vertical line at beta_c (2D Ising square lattice critical inverse temp)
beta_c = 0.4406868
ax.axvline(beta_c, color='b', linestyle='--', alpha=0.7, zorder=2)
ax.text(beta_c + 0.02, 0.6, r'$\beta_c$', color='b', fontsize=16)

# Horizontal line at log 2
ax.axhline(np.log(2), color='red', linestyle='--', linewidth=2, zorder=0)
ax.text(0.8, np.log(2) - 0.05, r'$\log 2$', color='red', fontsize=16)

# Make and save plot
fig.tight_layout()
fig.savefig('fig_asymmetry_deformed_tc.pdf', bbox_inches='tight')
plt.show()
