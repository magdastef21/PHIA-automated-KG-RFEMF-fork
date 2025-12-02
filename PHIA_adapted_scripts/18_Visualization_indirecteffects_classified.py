import pandas as pd
import os
import numpy as np

def assign_keys_to_uniq_var(df, columns, key_prefix):
    '''Assign unique keys to combinations of values in specified columns.'''
    df['combined_key'] = df[columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    uniq_val = sorted(set(df['combined_key'].dropna()))
    uniq_keys = [key_prefix + str(i) for i in range(1, len(uniq_val) + 1)]
    df[key_prefix + "keys"] = df['combined_key'].map(dict(zip(uniq_val, uniq_keys)))
    return df

def check_missing_nodes(nodes_df, edge_df):
    '''Check if all nodes are present in the node list.'''
    sources = set(edge_df['Source'])
    targets = set(edge_df['Target'])
    all_edge_nodes = sources.union(targets)
    node_ids = set(nodes_df['Id'])
    missing_nodes = all_edge_nodes - node_ids
    if missing_nodes:
        print(f"Warning: Missing {len(missing_nodes)} nodes in the node list. Adding them.")
        missing_nodes_df = pd.DataFrame({'Id': list(missing_nodes), 'Label': 'Unknown', 'Mode': 0})
        nodes_df = pd.concat([nodes_df, missing_nodes_df], ignore_index=True)
    return nodes_df

def prepare_colored_kg_with_ecological_linking(df, output_dir):
    '''Generate a KG linking Direct Effects to Ecological Consequences, then to Indirect Effects.'''
    # Prepare nodes: Direct Effects, Ecological Consequences, and Indirect Effects
    nodes = pd.concat([
        df[['DE_keys', 'DirectEffects']].rename(columns={'DE_keys': 'Id', 'DirectEffects': 'Label'}),
        df[['EC_keys', 'EcoConsequences', 'StudyGroup']].rename(columns={'EC_keys': 'Id', 'EcoConsequences': 'Label'}),
        df[['IE_keys', 'IndirectEffects', 'StudyGroup']].rename(columns={'IE_keys': 'Id', 'IndirectEffects': 'Label'})
    ], ignore_index=True)

    # Assign colors/modes
    nodes['Mode'] = 0  # Default mode for all
    nodes.loc[nodes['Id'].str.startswith('DE_'), 'Mode'] = 1  # Direct Effects
    nodes.loc[nodes['Id'].str.startswith('EC_'), 'Mode'] = df['StudyGroup'].factorize()[0] + 2  # Ecological Consequences based on Study Group
    nodes.loc[nodes['Id'].str.startswith('IE_'), 'Mode'] = df['StudyGroup'].factorize()[0] + 2  # Indirect Effects based on Study Group

    # Remove duplicates
    nodes = nodes.drop_duplicates()

    # Prepare edges
    edges = []
    for _, row in df.iterrows():
        counts_mention = row.get('counts_mention', 1)  # Default to 1 if not found
        if pd.notna(row.get('EcoConsequences')):  # If Ecological Consequences exist
            edges.append({'Source': row['DE_keys'], 'Target': row['EC_keys'], 'Weight': counts_mention})
            edges.append({'Source': row['EC_keys'], 'Target': row['IE_keys'], 'Weight': counts_mention})
        else:  # If no Ecological Consequences exist
            edges.append({'Source': row['DE_keys'], 'Target': row['IE_keys'], 'Weight': counts_mention})

    edges = pd.DataFrame(edges)

    # Check for missing nodes
    nodes = check_missing_nodes(nodes, edges)

    # Export to CSV
    nodes_file = os.path.join(output_dir, "colored_kg_nodes.csv")
    edges_file = os.path.join(output_dir, "colored_kg_edges.csv")
    nodes.to_csv(nodes_file, index=False)
    edges.to_csv(edges_file, index=False)
    print(f"Nodes file saved to {nodes_file}")
    print(f"Edges file saved to {edges_file}")

# Main Execution
# Set the directory to where your input file is located
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects")

# Load the input data
evidence_instances_full = pd.read_csv("unique_evidence_instances_true_clean_harm_final.csv")

# Clean the data
evidence_instances_full.replace({"-100": np.nan}, inplace=True)
evidence_instances_full.fillna(np.nan, inplace=True)

# Standardize text columns
evidence_instances_full['StudyGroup'] = evidence_instances_full['StudyGroup'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['IndirectEffects'] = evidence_instances_full['IndirectEffects'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['DirectEffects'] = evidence_instances_full['DirectEffects'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['EcoConsequences'] = evidence_instances_full['EcoConsequences'].astype(str).str.lower().replace("nan", "unknown")

# Assign unique keys
evidence_instances_full = assign_keys_to_uniq_var(evidence_instances_full, ["IndirectEffects", "StudyGroup"], "IE_")
evidence_instances_full = assign_keys_to_uniq_var(evidence_instances_full, ["DirectEffects"], "DE_")
evidence_instances_full = assign_keys_to_uniq_var(evidence_instances_full, ["EcoConsequences", "StudyGroup"], "EC_")

# Check assigned keys
print("\nAssigned Unique Keys for IndirectEffects, StudyGroup, and Ecological Consequences:")
print(evidence_instances_full[['IndirectEffects', 'StudyGroup', 'EcoConsequences', 'IE_keys', 'EC_keys']].drop_duplicates())

# Ensure counts columns exist
if 'counts_mention' not in evidence_instances_full.columns:
    evidence_instances_full['counts_mention'] = 1
if 'counts_articles' not in evidence_instances_full.columns:
    evidence_instances_full['counts_articles'] = 1

# Generate the KG files
output_directory = r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects"
os.makedirs(output_directory, exist_ok=True)

prepare_colored_kg_with_ecological_linking(evidence_instances_full, output_directory)
