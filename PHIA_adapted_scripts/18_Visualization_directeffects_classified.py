import pandas as pd
import numpy as np
import os

def assign_keys_to_uniq_var(df, column, key_prefix):
    ''' Assign unique keys to unique values in a column. '''
    uniq_val = sorted(set(df[column].dropna()))
    uniq_keys = [key_prefix + str(i) for i in range(1, len(uniq_val) + 1)]
    df[key_prefix + "keys"] = df[column].map(dict(zip(uniq_val, uniq_keys)))
    return df

def prepare_output_tables_Gephi_subgraph_ETHESG(df, subset_keys, graph_name):
    ''' Generate nodes and edge lists for the given subset of data. '''
    df = df.copy()  # Ensure we're working with a copy of the DataFrame
    df['ExposureType'] = df['ExposureType'].astype(str)
    
    # Convert subset_keys to strings and strip any whitespace
    subset_keys = [str(key).lower().strip() if pd.notna(key) else '' for key in subset_keys]
    
    # Create a regex pattern from the subset_keys
    subset_pattern = '|'.join(subset_keys)
    
    # Filter the DataFrame based on the pattern
    sub_graph_edgelist = df[df['ExposureType'].str.contains(subset_pattern, regex=True, na=False)]
    
    required_columns = ['ET_keys', 'HE_keys', 'SG_keys', 'ST_keys', 'SE_keys', 'MO_keys']
    missing_columns = [col for col in required_columns if col not in sub_graph_edgelist.columns]
    if missing_columns:
        print(f"Missing columns before processing: {missing_columns}")
        raise KeyError(f"Missing columns: {missing_columns}") #change here everything should be linked to HE
    
    sub_graph_edgelist1 = sub_graph_edgelist[['ET_keys', 'HE_keys', 'stat_knowledge', 'counts_mention', 'counts_articles']].rename(columns={'ET_keys': 'Source', 'HE_keys': 'Target'})
    sub_graph_edgelist2 = sub_graph_edgelist[['HE_keys', 'SG_keys', 'stat_knowledge', 'counts_mention', 'counts_articles']].rename(columns={'HE_keys': 'Source', 'SG_keys': 'Target'})
    sub_graph_edgelist3 = sub_graph_edgelist[['HE_keys', 'ST_keys', 'stat_knowledge', 'counts_mention', 'counts_articles']].rename(columns={'HE_keys': 'Source', 'ST_keys': 'Target'})
    sub_graph_edgelist4 = sub_graph_edgelist[['HE_keys', 'SE_keys', 'stat_knowledge', 'counts_mention', 'counts_articles']].rename(columns={'HE_keys': 'Source', 'SE_keys': 'Target'})
    sub_graph_edgelist5 = sub_graph_edgelist[['HE_keys', 'MO_keys', 'stat_knowledge', 'counts_mention', 'counts_articles']].rename(columns={'HE_keys': 'Source', 'MO_keys': 'Target'})
    
    nodes = pd.concat([
        sub_graph_edgelist[['ET_keys', 'ExposureType']].rename(columns={'ET_keys': 'Id', 'ExposureType': 'Label'}),
        sub_graph_edgelist[['HE_keys', 'HealthEffects']].rename(columns={'HE_keys': 'Id', 'HealthEffects': 'Label'}),
        sub_graph_edgelist[['SG_keys', 'StudyGroup']].rename(columns={'SG_keys': 'Id', 'StudyGroup': 'Label'}),
        sub_graph_edgelist[['ST_keys', 'StudyType']].rename(columns={'ST_keys': 'Id', 'StudyType': 'Label'}),
        sub_graph_edgelist[['SE_keys', 'StudyEnvironment']].rename(columns={'SE_keys': 'Id', 'StudyEnvironment': 'Label'}),
        sub_graph_edgelist[['MO_keys', 'Moderator']].rename(columns={'MO_keys': 'Id', 'Moderator': 'Label'})
    ], ignore_index=True, axis=0)
    
    nodes["Mode"] = 4
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['ET_keys']), "Mode"] = 2
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['HE_keys']), "Mode"] = 3
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['SG_keys']), "Mode"] = 5
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['ST_keys']), "Mode"] = 6
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['SE_keys']), "Mode"] = 7
    nodes.loc[nodes['Id'].isin(sub_graph_edgelist['MO_keys']), "Mode"] = 8
    nodes = nodes.drop_duplicates()
    
    superclass_df = pd.DataFrame({
        "Source": ['ET_class'] * len(set(sub_graph_edgelist['ET_keys'])),
        "Target": list(set(sub_graph_edgelist['ET_keys'])),
        "sign_consist": [3] * len(set(sub_graph_edgelist['ET_keys'])),
        "counts_mention": [100] * len(set(sub_graph_edgelist['ET_keys'])),
        "counts_articles": [100] * len(set(sub_graph_edgelist['ET_keys']))
    })
    
    sub_graph_edgelist = pd.concat([sub_graph_edgelist1, sub_graph_edgelist2, sub_graph_edgelist3, sub_graph_edgelist4, sub_graph_edgelist5, superclass_df], ignore_index=True, axis=0)
    
    sub_graph_edgelist.to_csv(f"{graph_name}_edgelist_ETHESG.csv", index=False)
    nodes.to_csv(f"{graph_name}_nodeslist_ETHESG.csv", index=False)

def prepare_output_tables_Gephi_for_all_studygroups(df):
    ''' Generate separate node- and edgelist files for each unique StudyGroup, plus a global one. '''
    unique_studygroups = df['StudyGroup'].unique()
    
    # For the global dataset (all study groups)
    graph_name_global = "Graph_All_StudyGroups"
    prepare_output_tables_Gephi_subgraph_ETHESG(
        df=df, 
        subset_keys=df['ExposureType'].unique(), 
        graph_name=graph_name_global
    )

    # For each unique study group
    for studygroup in unique_studygroups:
        subset_df = df[df['StudyGroup'] == studygroup].copy()  # Ensure we're working with a copy
        graph_name = f"Graph_{studygroup.replace(' ', '_')}"
        prepare_output_tables_Gephi_subgraph_ETHESG(
            df=subset_df, 
            subset_keys=subset_df['ExposureType'].unique(), 
            graph_name=graph_name
        )

# Execution

# Reading the evidence data
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg\classification")
evidence_instances_full = pd.read_csv("visualisation_dataset.csv", encoding='iso-8859-1')

# Preparing the data
evidence_instances_full.replace({"-100": np.nan}, inplace=True)
evidence_instances_full.fillna(np.nan, inplace=True)

# Standardize text data
evidence_instances_full['StudyGroup'] = evidence_instances_full['StudyGroup'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['StudyType'] = evidence_instances_full['StudyType'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['StudyEnvironment'] = evidence_instances_full['StudyEnvironment'].astype(str).str.lower().replace("nan", "unknown")
evidence_instances_full['Moderator'] = evidence_instances_full['Moderator'].astype(str).str.lower()

# Calculate sign consistency #change that --> evidence vs hypothesis
evidence_instances_full['stat_knowledge'] = np.where(
    (evidence_instances_full['stat_knowledge'].astype(str).str.contains("studied", case=False, na=False)),
    1, 0
)

# Initialize counts columns
evidence_instances_full['counts_mention'] = 0
evidence_instances_full['counts_articles'] = 0

# Unique edgelist preparation
uniq_edgelist = evidence_instances_full.drop_duplicates(subset=["ExposureType", "HealthEffects", "StudyGroup", "StudyType", "StudyEnvironment", "Moderator"]).copy()
uniq_edgelist = uniq_edgelist.reset_index(drop=True)

# Counting mentions and articles
for x in range(len(uniq_edgelist)):
    mentions = list(np.where(
        (evidence_instances_full["ExposureType"] == uniq_edgelist["ExposureType"].iloc[x]) &
        (evidence_instances_full["HealthEffects"] == uniq_edgelist["HealthEffects"].iloc[x]) &
        (evidence_instances_full["StudyGroup"] == uniq_edgelist["StudyGroup"].iloc[x]) &
        (evidence_instances_full["StudyType"] == uniq_edgelist["StudyType"].iloc[x]) &
        (evidence_instances_full["StudyEnvironment"] == uniq_edgelist["StudyEnvironment"].iloc[x]) &
        (evidence_instances_full["Moderator"] == uniq_edgelist["Moderator"].iloc[x])
    )[0])
    uniq_edgelist.at[x, "counts_mention"] = len(mentions)
    uniq_edgelist.at[x, "counts_articles"] = len(set(evidence_instances_full["DOI"].iloc[mentions]))

# Assign keys
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "ExposureType", "ET_")
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "HealthEffects", "HE_")
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "StudyGroup", "SG_")
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "StudyType", "ST_")
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "StudyEnvironment", "SE_")
uniq_edgelist = assign_keys_to_uniq_var(uniq_edgelist, "Moderator", "MO_")

# Final output to Gephi
prepare_output_tables_Gephi_for_all_studygroups(uniq_edgelist)
