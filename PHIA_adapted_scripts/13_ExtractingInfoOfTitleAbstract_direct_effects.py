import os
import pandas as pd
import numpy as np
from itertools import chain

pd.options.display.max_colwidth = 100000

######################################################################
## This script can be used to extract Information on Exposure Types
## and Studygroups of Analysis from the Title and Abstract of a Scientific artile
#################################################################################
#ET
#Functions
def extendInfofromTitleAbstract(articlesdata, fulllabeledtext, evidenceinstance_df):
    labeled_papers = list(dict.fromkeys(evidenceinstance_df['DOI']))
    evidenceinstance_df[['Title', 'Abstract', 'NrET_inArticle', 'NrTitleETappears', 'NrSG_inArticle', 'ExposureTypeInTitle', 'ExposureTypeInAbstract', 'StudyGroupInTitle', 'StudyGroupInAbstract']] = ""
    for file in labeled_papers:
        article_data_idx = list(np.where(articlesdata['filename'] == file)[0])
        evidence_data_idx = list(np.where(evidenceinstance_df['DOI'] == file)[0])
        labeled_words_idx = list(np.where(fulllabeledtext['filename'] == file)[0])
        title = articlesdata['title'].iloc[article_data_idx].to_string()
        evidenceinstance_df['Title'].iloc[evidence_data_idx] = title
        title = title.lower()
        ExposureTypes = fulllabeledtext['ExposureType'].iloc[labeled_words_idx]
        ExposureTypes = [elm.lower().split(" ; ") for elm in ExposureTypes if isinstance(elm, str)]
        ExposureTypes = list(dict.fromkeys(list(chain.from_iterable(ExposureTypes))))
        evidenceinstance_df['NrET_inArticle'].iloc[evidence_data_idx] = len(ExposureTypes)
        ET_inTitle = [elm for elm in ExposureTypes if title.find(elm) != -1]
        if len(ET_inTitle) > 1:
            longest_ET = max(ET_inTitle, key=len)
            if all(True for elm in ET_inTitle if elm in longest_ET):
                ET_inTitle = [longest_ET]
        if len(ET_inTitle) > 0:
            evidenceinstance_df['NrTitleETappears'].iloc[evidence_data_idx] = len([i for i in fulllabeledtext['ExposureType'].iloc[labeled_words_idx] if ET_inTitle[0] in str(i)])
        evidenceinstance_df['ExposureTypeInTitle'].iloc[evidence_data_idx] = " ; ".join(ET_inTitle)
        abstract = article_data['abstract'].iloc[article_data_idx].to_string().lower()
        evidenceinstance_df['Abstract'].iloc[evidence_data_idx] = abstract
        print(title)
        print(abstract)
        print(ET_inTitle)
        if ET_inTitle == []:
            ET_inAbstract = [elm for elm in ExposureTypes if abstract.find(elm) != -1]
            if len(ET_inAbstract) > 1:
                longest_ET = max(ET_inAbstract, key=len)
                if all(True for elm in ET_inAbstract if elm in longest_ET):
                    ET_inAbstract = [longest_ET]
            evidenceinstance_df['ExposureTypeInAbstract'].iloc[evidence_data_idx] = " ; ".join(ET_inAbstract)
            print(ET_inAbstract)
        StudyGroup = fulllabeledtext['StudyGroup'].iloc[labeled_words_idx]
        StudyGroup = [elm.lower().split(" ; ") for elm in StudyGroup if isinstance(elm, str)]
        StudyGroup = list(dict.fromkeys(list(chain.from_iterable(StudyGroup))))
        StudyGroup = [i for i in StudyGroup if i not in ["in", "of", "to"]]
        evidenceinstance_df['NrSG_inArticle'].iloc[evidence_data_idx] = len(StudyGroup)
        SG_inTitle = [elm for elm in StudyGroup if title.find(elm) != -1]
        if len(SG_inTitle) > 1:
            longest_SG = max(SG_inTitle, key=len)
            if all(True for elm in SG_inTitle if elm in longest_SG):
                SG_inTitle = [longest_SG]
        evidenceinstance_df['StudyGroupInTitle'].iloc[evidence_data_idx] = " ; ".join(SG_inTitle)
        print(SG_inTitle)
        if SG_inTitle == []:
            SG_inAbstract = [elm for elm in StudyGroup if abstract.find(elm) != -1]
            if len(SG_inAbstract) > 1:
                longest_SG = max(SG_inAbstract, key=len)
                if all(True for elm in SG_inAbstract if elm in longest_SG):
                    SG_inAbstract = [longest_SG]
            evidenceinstance_df['StudyGroupInAbstract'].iloc[evidence_data_idx] = " ; ".join(SG_inAbstract)
            print(SG_inAbstract)

    evidenceinstance_df['Title'] = [title.replace("Series([], )", "") for title in evidence_instances['Title']]
    return evidenceinstance_df


def find_lastET_for_missing_ET_info(fulllabeledtext, evidenceinstance_df):
    evidenceinstance_df[['last_ET', 'sent_last_ET', 'sent_distance_lastET']] = " "
    
    for count, value in enumerate(evidenceinstance_df['Sentence']):
        if str(evidenceinstance_df['ExposureType'].iloc[count]) == "-100":
            labeled_words_idx = list(np.where(fulllabeledtext['filename'] == evidenceinstance_df['DOI'].iloc[count])[0])
            current_idx = list(np.where((fulllabeledtext['filename'] == evidenceinstance_df['DOI'].iloc[count]) & 
                                        (fulllabeledtext['Sentence'] == evidenceinstance_df['Sentence'].iloc[count]))[0])
            
            if not labeled_words_idx or not current_idx:
                continue  # Skip if no indices are found

            if min(labeled_words_idx) > min(current_idx):
                continue  # Skip if indices are out of order

            potentialET = [i for i in fulllabeledtext['ExposureType'].iloc[min(labeled_words_idx): min(current_idx)] if str(i) != "nan"]
            sentenceID_ET = [val for cnt, val in enumerate(fulllabeledtext['Sentence'].iloc[min(labeled_words_idx):min(current_idx)]) if str(fulllabeledtext['ExposureType'].iloc[labeled_words_idx[cnt-1]]) != "nan"]
            
            if len(potentialET) > 0:
                evidenceinstance_df.loc[evidenceinstance_df.index[count], 'last_ET'] = potentialET[-1]
                evidenceinstance_df.loc[evidenceinstance_df.index[count], 'sent_last_ET'] = sentenceID_ET[-1]
                evidenceinstance_df.loc[evidenceinstance_df.index[count], 'sent_distance_lastET'] = int(evidenceinstance_df['Sentence'].iloc[count].replace("Sentence: ", "")) - int(evidenceinstance_df['sent_last_ET'].iloc[count].replace("Sentence: ", ""))
                print(potentialET[-1])
    return evidenceinstance_df


def select_correct_imputation(evidenceinstance_df):
    evidenceinstance_df[["ETimputed", "SGimputed"]] = ""
    for count, value in enumerate(evidenceinstance_df['ExposureType']):
        if value == "-100":
            if len(evidenceinstance_df['ExposureTypeInTitle'].iloc[count]) > 0:
                if ((evidenceinstance_df['ExposureTypeInTitle'].iloc[count] == evidenceinstance_df['last_ET'].iloc[count]) |
                    (evidenceinstance_df['ExposureTypeInTitle'].iloc[count] in evidenceinstance_df['Fullsentence'].iloc[count])):
                    evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['ExposureTypeInTitle'].iloc[count]
                    evidenceinstance_df["ETimputed"].iloc[count] = 1
                elif ((evidenceinstance_df['ExposureTypeInTitle'].iloc[count] in evidenceinstance_df['last_ET'].iloc[count]) & (evidenceinstance_df['sent_distance_lastET'].iloc[count] <3)):
                    if ";" in evidenceinstance_df['last_ET'].iloc[count]:
                        ETset = evidenceinstance_df['last_ET'].iloc[count].split(" ; ")
                        ETset = list(dict.fromkeys([i for i in ETset if evidenceinstance_df['ExposureTypeInTitle'].iloc[count] in i]))
                        if len(ETset) == 1:
                            evidenceinstance_df['ExposureType'].iloc[count] = ETset[0]
                    else:
                        evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['last_ET'].iloc[count]
                        evidenceinstance_df["ETimputed"].iloc[count] = 1
                elif ((evidenceinstance_df['last_ET'].iloc[count] in evidenceinstance_df['ExposureTypeInTitle'].iloc[count]) & (evidenceinstance_df['NrTitleETappears'].iloc[count] > 20)):
                    evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['ExposureTypeInTitle'].iloc[count]
                    evidenceinstance_df["ETimputed"].iloc[count] = 1
                elif ((evidenceinstance_df['NrTitleETappears'].iloc[count]  > evidenceinstance_df['NrET_inArticle'].iloc[count] )| (evidenceinstance_df['NrTitleETappears'].iloc[count] > 25)):
                    evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['ExposureTypeInTitle'].iloc[count]
                    evidenceinstance_df["ETimputed"].iloc[count] = 1
                elif evidenceinstance_df['sent_distance_lastET'].iloc[count] <3:
                    if ";" in evidenceinstance_df['last_ET'].iloc[count]:
                        ETset = evidenceinstance_df['last_ET'].iloc[count].split(" ; ")
                        print(ETset)
                        one = list(dict.fromkeys([i for i in ETset if (i in evidenceinstance_df['ExposureType'].iloc[count-1]) or (i in evidenceinstance_df['ExposureType'].iloc[count+1]) or (str(evidenceinstance_df['ExposureType'].iloc[count-1]) in i) or (str(evidenceinstance_df['ExposureType'].iloc[count+1]) in i)]))
                        print(one)
                        if len(one) == 1:
                            evidenceinstance_df['ExposureType'].iloc[count] = one[0]
                        elif len(one) > 1:
                            evidenceinstance_df['ExposureType'].iloc[count] = min(one, key=len)
                        else:
                            one = [i for i in one if any(True for x in one if x in i)]
                            if len(one) == 1:
                                evidenceinstance_df['ExposureType'].iloc[count] = one[0]
                            one = list(dict.fromkeys([i for i in ETset if i in evidenceinstance_df['ExposureType'].iloc[count-5:count+5]]))
                            if len(one) == 1:
                                evidenceinstance_df['ExposureType'].iloc[count] = one[0]
                    else:
                        evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['last_ET'].iloc[count]
                        evidenceinstance_df["ETimputed"].iloc[count] = 1
            else:
                if isinstance(evidenceinstance_df['sent_distance_lastET'].iloc[count], int):
                    if evidenceinstance_df['sent_distance_lastET'].iloc[count] < 3:
                        evidenceinstance_df['ExposureType'].iloc[count] = evidenceinstance_df['last_ET'].iloc[count]
                        evidenceinstance_df["ETimputed"].iloc[count] = 1

    for count, value in enumerate(evidenceinstance_df['StudyGroup']):
        if ((value == "-100") & (evidenceinstance_df['StudyGroupInTitle'].iloc[count] != "")):
            evidenceinstance_df['StudyGroup'].iloc[count] = evidenceinstance_df['StudyGroupInTitle'].iloc[count]
            evidenceinstance_df["SGimputed"].iloc[count] = 1

    return evidenceinstance_df


# Execution
os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg")
article_data = pd.read_csv("metareview_details.csv")
article_data['filename'] = [( doi.replace("/", "_")) for doi in article_data['doi']]

os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Direct_Effects\final_kg")
labeled_words = pd.read_csv("Evidence_instances_df.csv")
labeled_words['filename'] = [doi.replace(".csv", "").replace("doi_", "") for doi in labeled_words['DOI']]
evidence_instances = pd.read_csv("unique_evidence_instances.csv")
evidence_instances['DOI'] = [doi.replace(".csv", "").replace("doi_", "") for doi in evidence_instances['DOI']]


evidence_instances = extendInfofromTitleAbstract(articlesdata = article_data, fulllabeledtext = labeled_words, evidenceinstance_df= evidence_instances)
evidence_instances = find_lastET_for_missing_ET_info(fulllabeledtext = labeled_words, evidenceinstance_df= evidence_instances)
evidence_instances = select_correct_imputation(evidenceinstance_df = evidence_instances)

csv = os.path.join(os.getcwd(), ("unique_evidence_instances_interpolatingMissinginfo.csv"))
evidence_instances.to_csv(csv, index=False)

csv = os.path.join(os.getcwd(), ("unique_evidence_instances_clean.csv"))
evidence_instances.iloc[:, :11].to_csv(csv, index=False)