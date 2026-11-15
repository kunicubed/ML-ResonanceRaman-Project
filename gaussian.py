from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Files and constants

HARTREE_TO_CM = 219474.63
inputfile = "scan_results.csv"
outfil = ""
EXC_SHIFT_CM = -1200.0   # rigid shift of the excited surface in cm^-1 (negative = lower); tune to match experiment

# Parse
def readFile(inputfile): 
    text = Path(inputfile).read_text()
    coords, energies, mu_y = [],[],[]

    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 11:
            continue
        try:
            coords.append([float(parts[2]),float(parts[3])])
            energies.append([float(parts[6]),float(parts[7])])
            mu_y.append(float(parts[10]))
        except ValueError:
            continue
    
    return np.array(coords), np.array(energies), np.array(mu_y)

# Define a function for fitting the surface

def fitsurface(coords, energies, mu_y):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[3.0, 3.0], length_scale_bounds=(0.3, 30.0))+ WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-4)))
    gp_ground = GaussianProcessRegressor(kernel = kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    gp_excited = GaussianProcessRegressor(kernel = kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    gp_trans = GaussianProcessRegressor(kernel = kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    
    groundenergies = energies[:, 0]
    excitedenergies = energies[:, 1]
    
    fittedGround = gp_ground.fit(coords, groundenergies)   
    fittedExcited = gp_excited.fit(coords, excitedenergies)
    fittedTrans = gp_trans.fit(coords, mu_y)
    
    return fittedGround, fittedExcited, fittedTrans

def cross_validate(coords, values, n_splits=5):
    pred = np.zeros(len(values))
    std = np.zeros(len(values))
    
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[3.0, 3.0], length_scale_bounds=(0.3, 30.0))+ WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-4)))
    gaussian_process = GaussianProcessRegressor(kernel = kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_rows, test_rows in folds.split(coords):
        gp = gaussian_process.fit(coords[train_rows], values[train_rows])
        pred[test_rows], std[test_rows] = gp.predict(coords[test_rows], return_std=True)
    return pred, std
    
coords, energies, mu_y = readFile(inputfile)
keep = ~((coords[:, 0] > 10) & (coords[:, 1] > 6))
coords, energies, mu_y = coords[keep], energies[keep], mu_y[keep]
energies = (energies - energies[:,0].min()) * HARTREE_TO_CM

def error_stats(pred, true, mask=None):
    err = pred - true
    if mask is not None: err = err[mask]
    return np.sqrt(np.mean(err**2)), np.max(np.abs(err))

inner = (coords[:,0] >= -2) & (coords[:,0] <= 10) & (coords[:,1] >= -2) & (coords[:,1] <= 5)

gp_g, gp_e, gp_t = fitsurface(coords, energies, mu_y)
print(gp_g.kernel_, gp_e.kernel_, gp_t.kernel_)

qs = np.linspace(-3, 10, 131)
qb = np.linspace(-3, 6, 91)
QS, QB = np.meshgrid(qs, qb, indexing="ij")
pts = np.column_stack([QS.ravel(), QB.ravel()])
Vg = gp_g.predict(pts).reshape(QS.shape)
Ve = gp_e.predict(pts).reshape(QS.shape)
Mu = gp_t.predict(pts).reshape(QS.shape)

print(Vg.min())                      
i = np.unravel_index(Ve.argmin(), Ve.shape)
print(qs[i[0]], qb[i[1]], Ve.min())   

w_s, w_b = 964.5/HARTREE_TO_CM, 455.4/HARTREE_TO_CM    # hartree
Qs, Qb = QS/np.sqrt(w_s), QB/np.sqrt(w_b)

E0 = Vg.min(); Vg -= E0; Ve -= E0
Ve += EXC_SHIFT_CM                     # rigid shift of the excited surface to match experiment
print("excited minimum after shift:", Ve.min())


np.savetxt("pot_ground.dat", np.column_stack([Qs.ravel(), Qb.ravel(), Vg.ravel()/HARTREE_TO_CM]))
np.savetxt("pot_excited.dat", np.column_stack([Qs.ravel(), Qb.ravel(), Ve.ravel()/HARTREE_TO_CM]))
np.savetxt("dipole.dat", np.column_stack([Qs.ravel(), Qb.ravel(), Mu.ravel()]))