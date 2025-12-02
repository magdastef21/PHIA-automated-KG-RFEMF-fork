import pandas as pd
import os
import numpy as np

def assign_keys_to_uniq_var(df, columns, key_prefix):
    ''' Assign unique keys to combinations of values in specified columns. '''
    # Create a new column that combines the relevant columns (e.g., HealthEffects + StudyGroup)
    df['combined_key'] = df[columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # Generate unique keys based on the combined column
    uniq_val = sorted(set(df['combined_key'].dropna()))
    uniq_keys = [key_prefix + str(i) for i in range(1, len(uniq_val) + 1)]
    
    # Assign the unique keys to the dataframe
    df[key_prefix + "keys"] = df['combined_key'].map(dict(zip(uniq_val, uniq_keys)))
    return df

def check_missing_nodes(nodes_df, edge_df):
    """
    Check if all nodes are present in the node list.
    """
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

def prepare_colored_kg_by_studygroup(df, output_dir):
    """
    Generate a KG with Exposure Types connected to Health Effects,
    and Health Effects colored based on Study Group.
    """
    # Prepare nodes: Exposure Types and Health Effects
    nodes = pd.concat([ 
        df[['ET_keys', 'ExposureType']].rename(columns={'ET_keys': 'Id', 'ExposureType': 'Label'}), 
        df[['HE_keys', 'HealthEffects', 'StudyGroup']].rename(columns={'HE_keys': 'Id', 'HealthEffects': 'Label'})
    ], ignore_index=True)

    # Assign colors/modes for study groups
    nodes['Mode'] = 1  # Default for Exposure Types
    nodes.loc[nodes['Id'].str.startswith('HE_'), 'Mode'] = df['StudyGroup'].factorize()[0] + 2

    # Remove duplicates
    nodes = nodes.drop_duplicates()

    # Prepare edges: Connect Exposure Types to Health Effects
    edges = df[['ET_keys', 'HE_keys', 'counts_mention', 'counts_articles']].rename(
        columns={'ET_keys': 'Source', 'HE_keys': 'Target'}
    )
    edges['Weight'] = edges['counts_mention']  # Optional: Use mentions as edge weights

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
# Change the directory to where your input file is located
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg\classification")

# Load the input data
evidence_instances_full = pd.read_csv("visualisation_dataset.csv", encoding='iso-8859-1')

# Check the initial structure of the data
print("Initial Data Structure:")
print(evidence_instances_full.head())  # Check the first few rows
print("\nColumns in the dataset:")
print(evidence_instances_full.columns)

# Clean the data
evidence_instances_full.replace({"-100": np.nan}, inplace=True)
evidence_instances_full.fillna(np.nan, inplace=True)

# Standardize text columns
evidence_instances_full['StudyGroup'] = evidence_instances_full['StudyGroup'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['HealthEffects'] = evidence_instances_full['HealthEffects'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['ExposureType'] = evidence_instances_full['ExposureType'].astype(str).str.lower().replace("nan", "unknown")

# Assign unique keys for each category (HealthEffects + StudyGroup for unique combination)
evidence_instances_full = assign_keys_to_uniq_var(evidence_instances_full, ["HealthEffects", "StudyGroup"], "HE_")

# Assign unique keys for ExposureType
evidence_instances_full = assign_keys_to_uniq_var(evidence_instances_full, ["ExposureType"], "ET_")

# Check assigned keys
print("\nAssigned Unique Keys for HealthEffects and StudyGroup:")
print(evidence_instances_full[['HealthEffects', 'StudyGroup', 'HE_keys']].drop_duplicates())

# Initialize counts columns if not present
if 'counts_mention' not in evidence_instances_full.columns:
    evidence_instances_full['counts_mention'] = 1
if 'counts_articles' not in evidence_instances_full.columns:
    evidence_instances_full['counts_articles'] = 1

# Generate the KG files
output_directory = r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg\classification"
os.makedirs(output_directory, exist_ok=True)

prepare_colored_kg_by_studygroup(evidence_instances_full, output_directory)

# Load the dataset again to check structure
file_path = r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg\classification\visualisation_dataset.csv"
evidence_instances_full = pd.read_csv(file_path, encoding='iso-8859-1')
