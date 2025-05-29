import numpy as np
from scipy.linalg import inv, det

def h_index(cites):
    cites_sorted = sorted(cites, reverse=True)
    return max((i+1) for i,c in enumerate(cites_sorted) if c>=i+1) if cites_sorted and cites_sorted[0]>0 else 0

def kl_gauss(m0, S0, m1, S1):
    """KL( N0 || N1 ) closed‑form."""
    d = m0.shape[0]
    S1_inv = inv(S1)
    term1 = np.trace(S1_inv @ S0)
    diff  = (m1 - m0).reshape(-1,1)
    term2 = float(diff.T @ S1_inv @ diff)
    term3 = np.log(det(S1) / max(det(S0), 1e-10))
    return 0.5 * (term1 + term2 - d + term3)

def compute_density_score(vars_set, variable_sets_seen, mu, S, kl_scale):
    if vars_set not in variable_sets_seen:
        dens_score = 1.0
    else:
        kls = [
            kl_gauss(mu, S, mu_prev, S_prev)
            for (mu_prev, S_prev) in variable_sets_seen[vars_set]
        ]
        avg_kl = np.mean(kls)
        dens_score = min(1.0, avg_kl / kl_scale)
    return dens_score

def calculate_scrud(ag, datasets, citations):
    cites_u = [citations[d] for d in ag["datasets"] if datasets[d]["ctx"]=="uncontrolled"]
    cites_c = [citations[d] for d in ag["datasets"] if datasets[d]["ctx"]=="controlled"]
    dens_u  = [datasets[d]["density"] for d in ag["datasets"] if datasets[d]["ctx"]=="uncontrolled"]
    dens_c  = [datasets[d]["density"] for d in ag["datasets"] if datasets[d]["ctx"]=="controlled"]
    
    Ru = h_index(cites_u); Rc = h_index(cites_c)
    Ru_s = Ru / max(Ru, Rc, 1); Rc_s = Rc / max(Ru, Rc, 1)  # crude scaling
    Du = np.mean(dens_u) if dens_u else 0.0
    Dc = np.mean(dens_c) if dens_c else 0.0

    # Use a dynamic harmonic mean to calculate the SCRUD index because 
    # You don't want to penalize agents that work in controlled or uncontrolled contexts
    values = []
    if Ru > 0: values.append(Ru_s)
    if Rc > 0: values.append(Rc_s)
    if Du > 0:  values.append(Du)
    if Dc > 0:  values.append(Dc)

    if values:
        hmn = (np.prod(values) + 1e-12) ** (1.0 / len(values))
    else:
        hmn = -1
    
    scrud = hmn * len(ag["datasets"])
    return scrud, hmn, Ru, Rc, Du, Dc