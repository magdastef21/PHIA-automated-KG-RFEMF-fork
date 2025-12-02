import os
import pandas as pd

def classify_into_categories(text):
    """Classify the text into updated categories based on keywords."""
    # Define category keywords
    categories = {
        "knowledge": ["addressed", "analysed", "assessed", "documented", "evaluated", 
                      "examined", "explored", "focus", "hypothesis", "investigated", 
                      "involvement", "looked", "may", "might", "monitored", "performed", 
                      "possible", "potential", "reviewed", "seen", "sought", "studied", 
                      "tested", "indicated"],
        "relationship": ["not", "non", "absence", "no", "lack of", "unlikely", "failed", 
                         "neither", "nor", "without"],
        "direction": ["promote", "upregulate", "increase", "elevated", "facilitating", 
                      "positive", "rise", "higher", "activates", "enhance", "more", 
                      "increment", "improved", "beneficial", "buildup", "dilates", 
                      "maximum", "superior"],
        "consistency": ["inconsistent", "controversial", "inadequate", "no consistency", 
                        "mixed", "unclear", "contradictory", "insufficient", "no convincing"]
    }
    
    knowledge = "evidence"  # Default to "evidence" if not matched
    statrelat = "relationship"
    statdirect = "decrease/negative"
    statconsist = "consistent"
    statother = "other"

    # Check for knowledge category
    if any(keyword in text for keyword in categories["knowledge"]):
        knowledge = "hypothesis"
    
    # Check for relationship category
    if knowledge == "evidence" and any(keyword in text for keyword in categories["relationship"]):
        statrelat = "no relationship"
    
    # Check for direction category
    if knowledge == "evidence" and any(keyword in text for keyword in categories["direction"]):
        statdirect = "increase/positive"
    
    # Check for consistency category
    if knowledge == "evidence" and any(keyword in text for keyword in categories["consistency"]):
        statconsist = "inconsistent"
    
    # Return the categories
    return knowledge, statrelat, statdirect, statconsist, statother

# Set working directory and read CSV
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg")
csv_path = os.path.join(os.getcwd(), "predicted_evidence_instances_true.csv")
evidence_instances = pd.read_csv(csv_path)

# Select relevant columns
columns_of_interest = ['DOI', 'Sentence', 'Fullsentence', 'ExposureType', 'HealthEffects', 
                        'AssociationType', 'StudyGroup', 'Moderator', 'StudyType', 'StudyEnvironment']
evidence_instances = evidence_instances[columns_of_interest]

# Initialize new columns for classification results
evidence_instances["knowledge"] = ""
evidence_instances["stat_relationship"] = ""
evidence_instances["stat_direction"] = ""
evidence_instances["stat_consistency"] = ""
evidence_instances["stat_other"] = ""

# Classify each instance
for index, row in evidence_instances.iterrows():
    association_type = str(row['AssociationType']).lower() if pd.notnull(row['AssociationType']) else ""
    knowledge, statrelat, statdirect, statconsist, statother = classify_into_categories(association_type)
    evidence_instances.at[index, "knowledge"] = knowledge
    if knowledge == "evidence":
        evidence_instances.at[index, "stat_relationship"] = statrelat
        evidence_instances.at[index, "stat_direction"] = statdirect
        evidence_instances.at[index, "stat_consistency"] = statconsist
    evidence_instances.at[index, "stat_other"] = statother
    print("Result:", knowledge, statrelat, statdirect, statconsist, statother, row['AssociationType'])

# Save classified instances
evidence_instances.to_csv("classified_evidence_instances.csv", index=False)

# Print statistics and drop duplicates
print("Number of evidence instances:", len(evidence_instances))
evidence_instances.drop_duplicates(inplace=True)
print("Number of unique evidence instances:", len(evidence_instances))

# Determine completeness
evidence_instances["complete"] = evidence_instances[['knowledge', 'stat_relationship', 'stat_direction', 'stat_consistency', 'stat_other']].notna().all(axis=1).astype(int)

# Filter complete records
evidence_instances = evidence_instances[evidence_instances["complete"] == 1]
evidence_instances.drop(columns=["AssociationType", "complete"], inplace=True)
evidence_instances.drop_duplicates(inplace=True)

print("Number of contributing articles:", len(set(evidence_instances['DOI'])))
print("Number of unique and complete evidence instances:", len(evidence_instances))

# Save unique and complete instances
evidence_instances.to_csv("unique_evidence_instances.csv", index=False)
