import random

# Agent Profiles
PRODUCER_AGENT_PROFILES = {
    "High_Density": 0.8,
    "Random_Density": 0.5,
    "Low_Density": 0.2,
}

CONSUMER_AGENT_PROFILES = {
    "Uncontrolled_Preference": 0.8,
    "No_Preference": 0.5,
}

def setup_agents(rng, args):
    producer_agents = []
    consumer_agents = []
    n_high_density = int(args.n_producer_agents * args.high_density_ratio)
    n_random_density = int(args.n_producer_agents * args.random_density_ratio)
    n_low_density = args.n_producer_agents - n_high_density - n_random_density
    n_uncontrolled_preference = int(args.n_consumer_agents * args.uncontrolled_preference_ratio)
    n_no_preference = args.n_consumer_agents - n_uncontrolled_preference

    for i in range(n_high_density):
        producer_agents.append(dict(profile="High_Density", novel_variable_prob=PRODUCER_AGENT_PROFILES["High_Density"], datasets=[]))
    for i in range(n_random_density):
        producer_agents.append(dict(profile="Random_Density", novel_variable_prob=PRODUCER_AGENT_PROFILES["Random_Density"], datasets=[]))
    for i in range(n_low_density):
        producer_agents.append(dict(profile="Low_Density", novel_variable_prob=PRODUCER_AGENT_PROFILES["Low_Density"], datasets=[]))
    
    for i in range(n_uncontrolled_preference):
        consumer_agents.append(
            dict(
                profile="Uncontrolled_Preference", 
                preference_prob=CONSUMER_AGENT_PROFILES["Uncontrolled_Preference"],
                variables_of_interest=rng.choice(args.m, size=random.randint(1, 4), replace=False)
            )
        )
    for i in range(n_no_preference):
        consumer_agents.append(
            dict(
                profile="No_Preference", 
                preference_prob=CONSUMER_AGENT_PROFILES["No_Preference"],
                variables_of_interest=rng.choice(args.m, size=random.randint(1, 4), replace=False)
            )
        )
    
    return producer_agents, consumer_agents
