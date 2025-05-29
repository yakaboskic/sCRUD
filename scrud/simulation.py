import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scrud.graph import random_dag, sample_sem
from scrud.index import compute_density_score, calculate_scrud
from scrud.agents import setup_agents
from collections import defaultdict
from uuid import uuid4
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SCRUD Simulation Configuration")
    parser.add_argument("--m", type=int, default=15, help="Variables in the causal DAG")
    parser.add_argument("--edge_prob", type=float, default=0.25, help="Sparsity of the DAG")
    parser.add_argument("--n_runs", type=int, default=3, help="How many Monte-Carlo repetitions")
    parser.add_argument("--n_producer_agents", type=int, default=50, help="Number of producer agents")
    parser.add_argument("--high_density_ratio", type=float, default=0.1, help="Ratio of high density agents")
    parser.add_argument("--random_density_ratio", type=float, default=0.6, help="Ratio of random density agents")
    parser.add_argument("--low_density_ratio", type=float, default=0.3, help="Ratio of low density agents")
    parser.add_argument("--n_consumer_agents", type=int, default=50, help="Number of consumer agents")
    parser.add_argument("--uncontrolled_preference_ratio", type=float, default=0.5, help="Ratio of uncontrolled preference agents")
    parser.add_argument("--n_samples_data", type=int, default=120, help="Rows per dataset")
    parser.add_argument("--kl_scale", type=float, default=5.0, help="KL / KL_SCALE -> clip to 1 for density score")
    parser.add_argument("--turn_batch", type=int, default=10, help="Agents acting per turn")
    parser.add_argument("--stop_datasets", type=int, default=50, help="Stop when an agent reaches this")
    parser.add_argument("--lambda_high", type=float, default=8, help="Lambda for high agents")
    parser.add_argument("--lambda_low", type=float, default=2, help="Lambda for low agents")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--num_data_consumption_loop_cycles", type=int, default=10, help="Number of data consumption loop cycles per turn")
    return parser.parse_args()

# def sample_variable_set(rng, novel_variable_prob, variable_sample_count, k):
#     print(variable_sample_count)
#     _variable_sample_count = variable_sample_count.copy()
#     var_set = set()
#     while True:
#         # Choose top least explored variable with probability novel_variable_prob, otherwise random
#         if rng.random() < novel_variable_prob:
#             variable = min(_variable_sample_count, key=_variable_sample_count.get)
#         else:
#             variable = rng.choice(list(_variable_sample_count.keys()))
#         var_set.add(variable)
#         _variable_sample_count.pop(variable)
#         if len(var_set) == k:
#             break
#     print(f'Chosen variables: {var_set}')
#     return frozenset(var_set)

def sample_variable_set(rng, novel_variable_prob, variable_sample_count, variable_sets_seen, k):
    _variable_sample_count = variable_sample_count.copy()

    if rng.random() < novel_variable_prob:
        # Try to create a novel variable set
        attempts = 0
        while attempts < 50:
            sorted_vars = sorted(_variable_sample_count, key=_variable_sample_count.get)
            candidate = frozenset(rng.choice(sorted_vars[:len(sorted_vars)//2], size=k, replace=False))
            if candidate not in variable_sets_seen:
                return candidate
            attempts += 1
        # Fallback if all are exhausted
        return candidate
    else:
        # Choose a redundant set from those already seen
        if variable_sets_seen:
            return rng.choice(list(variable_sets_seen.keys()))
        else:
            # fallback to random if nothing has been seen yet
            return frozenset(rng.choice(list(_variable_sample_count.keys()), size=k, replace=False))


def run_one_sim(args):
    rng = np.random.default_rng(args.seed)
    logger.info(f"Running simulation with seed {args.seed}")
    A = random_dag(args.m, args.edge_prob, rng)
    logger.info(f"Created causal graph with {args.m} variables and {args.edge_prob} edge probability")
    
    # pre‑generate a big data matrix for reuse efficiency
    MASTER_DATA = sample_sem(A, 50000, rng)
    logger.info(f"Master data shape: {MASTER_DATA.shape}")
    
    # state containers
    variable_sets_seen = {}  # key=frozenset(vars) -> list of (mean,cov) tuples
    
    logger.info("Setting up agents")
    producer_agents, consumer_agents = setup_agents(rng, args)
    logger.info(f"Producer agents: {len(producer_agents)}")
    logger.info(f"Consumer agents: {len(consumer_agents)}")

    # Now assign random weights to each agent to have a spread of productivity in data set sharing for the simulation
    # Draw from a poisson distribution with mean 10
    weights = np.random.poisson(10, len(producer_agents))
    # Normalise weights to sum to 1
    weights = weights / weights.sum()
    # Assign weights to agents
    
    citations = defaultdict(int)
    datasets = defaultdict(dict)
    variable_sample_count = {i: 0 for i in range(args.m)}
    logger.info("Starting simulation")

    pbar = tqdm(total=args.stop_datasets, desc="Running simulation", leave=False)
    while max(len(a["datasets"]) for a in producer_agents) < args.stop_datasets:
        pbar.n = max(len(a["datasets"]) for a in producer_agents)
        pbar.refresh()
        acting_idx = rng.choice(len(producer_agents), size=args.turn_batch, replace=False, p=weights)
        # Data Production Loop
        for idx in acting_idx:
            ag = producer_agents[idx]
            # choose variable set based on there preference for choosing variables that are least explored
            k = rng.integers(3, 6)      # size of subset
            novel_variable_prob = ag["novel_variable_prob"]
            # Sample a variable set based on the novel_variable_prob
            vars_set = sample_variable_set(rng, novel_variable_prob, variable_sample_count, variable_sets_seen, k)
            # Update variable sample count
            for var in vars_set:
                variable_sample_count[var] += 1
            # pull rows from MASTER_DATA (simple random sample)
            rows_idx = rng.choice(MASTER_DATA.shape[0], size=args.n_samples_data, replace=False)
            data_subset = MASTER_DATA[rows_idx][:, list(vars_set)]
            mu = data_subset.mean(axis=0)
            dim = data_subset.shape[1]
            S = np.cov(data_subset, rowvar=False) + 1e-6 * np.eye(dim)
            
            # compute density score
            density_score = compute_density_score(vars_set, variable_sets_seen, mu, S, args.kl_scale)
            # store representation for future density calcs
            variable_sets_seen.setdefault(vars_set, []).append((mu, S))
            
            # context
            ctx = "controlled" if rng.random() < 0.5 else "uncontrolled"
            dataset_id = str(uuid4())
            datasets[dataset_id] = dict(
                id=dataset_id,
                vars=vars_set,
                density=density_score,
                ctx=ctx
            )
            logger.debug(f"Produced dataset {dataset_id} with density {density_score} and context {ctx}")
            ag["datasets"].append(dataset_id)
            logger.debug(f"Producer agent {ag['profile']} now has {len(ag['datasets'])} datasets")

        # Data Consumption Loop
        for _ in range(args.num_data_consumption_loop_cycles):
            for ag in consumer_agents:
                preference_prob = ag["preference_prob"]
                variables_of_interest = ag["variables_of_interest"]
                # Get all datasets that contain all variables of interest
                relevant_datasets = [d for d in datasets.values() if set(variables_of_interest).issubset(d["vars"])]
                if len(relevant_datasets) == 0:
                    continue
                relevant_datasets.sort(key=lambda x: x["density"], reverse=True)
                # Choose dataset based on preference probability
                if rng.random() < preference_prob:
                    # Then choose the dataset with the highest density score that is not contolled
                    for d in relevant_datasets:
                        if d["ctx"] == "uncontrolled":
                            dataset_id = d["id"]
                            break
                else:
                    dataset_id = relevant_datasets[0]["id"]
                # Update dataset citation count
                citations[dataset_id] += 1

    pbar.close()
    # Summarise agents
    rows = []
    for ag in producer_agents:
        scrud, hmn, Ru, Rc, Du, Dc = calculate_scrud(ag, datasets, citations)
        
        rows.append(
            dict(
                profile=ag["profile"],
                datasets=len(ag["datasets"]),
                sCRUD=scrud,
                hmn=hmn,
                Ru=Ru,
                Rc=Rc,
                Du=Du,
                Dc=Dc
            )
        )
    citations_df = pd.DataFrame(citations.items(), columns=["dataset_id", "citations"])
    return pd.DataFrame(rows)

def plot_results(df_all, args):
    regression_coefs = {}
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pretty-print mapping
    pretty_names = {
        "High_Density": "High Density",
        "Random_Density": "Random Density",
        "Low_Density": "Low Density"
    }

    # Color and marker maps (optional custom styling)
    marker_map = {
        "High Density": "o",
        "Random Density": "s",
        "Low Density": "D"
    }

    colors = plt.cm.Set2.colors  # color palette

    for i, (prof, grp) in enumerate(df_all.groupby("profile")):
        pretty = pretty_names.get(prof, prof.replace("_", " "))
        X = grp["datasets"].values.reshape(-1, 1)
        y = grp["sCRUD"].values

        # Plot scatter
        ax.scatter(X, y, alpha=0.6, label=pretty, marker=marker_map.get(pretty, "o"), color=colors[i % len(colors)])

        # Fit and plot linear regression line
        model = LinearRegression()
        model.fit(X, y)
        x_line = np.linspace(0, args.stop_datasets, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        ax.plot(x_line, y_line, linestyle="--", linewidth=2, color=colors[i % len(colors)], label=f"{pretty} Trend")

        X_ = sm.add_constant(X)  # adds intercept term
        model = sm.OLS(y, X_).fit()
        with open(f"scrud_simulation_results_{args.m}vars_{args.edge_prob}edge_prob_{args.n_runs}runs.txt", "a") as f:
            f.write(f"{prof} Regression coefficients: {model.params[1]}\n")
            f.write(f"{prof} Regression p-value: {model.pvalues[1]}\n")
            f.write(model.summary().as_text())
            f.write("\n")

    # Theoretical best
    ax.plot([0, args.stop_datasets], [0, args.stop_datasets], "--", lw=2, color="black", label="Theoretical best")

    ax.set_xlabel("# datasets shared")
    ax.set_ylabel("Scaled sCRUD")
    ax.set_title(f"sCRUD after {args.n_runs} Monte‑Carlo runs\n(M={args.m} vars, turn size={args.turn_batch})")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout() 
    plt.savefig(f"scrud_simulation_results_{args.m}vars_{args.edge_prob}edge_prob_{args.n_runs}runs.png")


def run_simulation(args):
    results = []
    for run in range(args.n_runs):
        logger.info(f"Running simulation run {run + 1} of {args.n_runs}")
        df_run = run_one_sim(args)
        df_run["run"] = run + 1
        results.append(df_run)
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Saving results to scrud_simulation_results_{args.m}vars_{args.edge_prob}edge_prob_{args.n_runs}runs.csv")
    df_all.to_csv(f"scrud_simulation_results_{args.m}vars_{args.edge_prob}edge_prob_{args.n_runs}runs.csv", index=False)
    plot_results(df_all, args)

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info(f"Running simulation with args: {args}")
    run_simulation(args)

if __name__ == "__main__":
    main()