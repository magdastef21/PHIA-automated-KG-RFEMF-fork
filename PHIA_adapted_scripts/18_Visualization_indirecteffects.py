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

def prepare_separate_graphs_for_direct_effects(df, output_dir):
    '''Generate a separate KG for each DirectEffect, classifying Direct, Indirect Effects, and Ecological Consequences by StudyGroup.'''
    
    # Get unique DirectEffects
    unique_direct_effects = df['DirectEffects'].drop_duplicates()

    # Loop through each DirectEffect and generate a separate subgraph
    for de in unique_direct_effects:
        
        # Filter the DataFrame for rows with the current DirectEffect
        df_de = df[df['DirectEffects'] == de]
        
        # Assign unique keys for DirectEffects, IndirectEffects, and EcologicalConsequences by combining with StudyGroup
        df_de = assign_keys_to_uniq_var(df_de, ['DirectEffects', 'StudyGroup'], 'DE_')
        df_de = assign_keys_to_uniq_var(df_de, ['IndirectEffects', 'StudyGroup'], 'IE_')
        df_de = assign_keys_to_uniq_var(df_de, ['EcoConsequences', 'StudyGroup'], 'EC_')
        
        # Prepare nodes: Direct Effects, Ecological Consequences, and Indirect Effects
        nodes = pd.concat([
            df_de[['DE_keys', 'DirectEffects', 'StudyGroup']].rename(columns={'DE_keys': 'Id', 'DirectEffects': 'Label'}),
            df_de[['EC_keys', 'EcoConsequences', 'StudyGroup']].rename(columns={'EC_keys': 'Id', 'EcoConsequences': 'Label'}),
            df_de[['IE_keys', 'IndirectEffects', 'StudyGroup']].rename(columns={'IE_keys': 'Id', 'IndirectEffects': 'Label'})
        ], ignore_index=True)
        
        # Classify Direct, Indirect Effects, and Ecological Consequences by StudyGroup and assign different colors/modes
        nodes['Mode'] = 0  # Default mode for all
        nodes.loc[nodes['Id'].str.startswith('DE_'), 'Mode'] = nodes['StudyGroup'].apply(lambda x: hash(x) % 10 + 1)  # Direct Effects
        nodes.loc[nodes['Id'].str.startswith('EC_'), 'Mode'] = nodes['StudyGroup'].apply(lambda x: hash(x) % 10 + 2)  # Ecological Consequences
        nodes.loc[nodes['Id'].str.startswith('IE_'), 'Mode'] = nodes['StudyGroup'].apply(lambda x: hash(x) % 10 + 3)  # Indirect Effects
        
        # Remove duplicates
        nodes = nodes.drop_duplicates()

        # Prepare edges
        edges = []
        for _, row in df_de.iterrows():
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
        nodes_file = os.path.join(output_dir, f"colored_kg_nodes_{de}.csv")
        edges_file = os.path.join(output_dir, f"colored_kg_edges_{de}.csv")
        nodes.to_csv(nodes_file, index=False)
        edges.to_csv(edges_file, index=False)
        print(f"Nodes file for {de} saved to {nodes_file}")
        print(f"Edges file for {de} saved to {edges_file}")

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

# Ensure counts columns exist
if 'counts_mention' not in evidence_instances_full.columns:
    evidence_instances_full['counts_mention'] = 1
if 'counts_articles' not in evidence_instances_full.columns:
    evidence_instances_full['counts_articles'] = 1

# Generate the KG files
output_directory = r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects"
os.makedirs(output_directory, exist_ok=True)

prepare_separate_graphs_for_direct_effects(evidence_instances_full, output_directory)



