import os
import spacy
en = spacy.load('en_core_web_sm')
from spacy import displacy
import networkx as nx
from itertools import chain
import statistics
import re
import pandas as pd
import numpy as np

################################################################################################
## This script creates syntactical properties of the labeled sentences to infer which phrase
## relates to which other to create an evidence instance. The property enriched dataframe can
## then be used to train a feature based machine learning model.
################################################################################################

# Functions
def SelectGrammCompleteSent(df):
    """tests which sentences contain at least a verb as a minimum grammatical requirement for a complete
       sentence and then subselects only these sentences."""
    gramm_complete_sentences = [x for i in ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'] for x, y in
                                enumerate(df['sentence_POS']) if i in y]
    df = df.iloc[gramm_complete_sentences]
    return df

def FixBracketsInVariableNames(df):
    """Remove brackets from phrase names of association types and behavior determinants"""
    df['AssociationType'] = [x.replace("; ( ", "; ").replace(" ) ;", " ;").replace(" ) ", " ; ").replace(" ( ", " ; ").replace("( ","").replace(
            " )", "").replace("(", "").replace(")", "").replace("  ", " ") for x in df['AssociationType']]
    df['HealthEffects'] = [x.replace("; ( ", "; ").replace(" ) ;", " ;").replace(" ) ", " ; ").replace(" ( ", " ; ").replace("( ","").replace(
            " )", "").replace("(", "").replace(")", "").replace("  ", " ").replace(" +", "") for x in df['HealthEffects']]
    return df

def replace_spacestr_NA (list):
    """Replace a list with only an empty space with an empty list."""
    if list == [" "]:
        newlist = []
    else:
        newlist = list
    return newlist

def createEntityList(df):
    """Split the listed phrases per sentences into entities per Label"""
    AT_entities = df['AssociationType'].iloc[count].split(" ; ")
    HE_entities = df['HealthEffects'].iloc[count].split(" ; ")
    ET_entities = replace_spacestr_NA(df['ExposureType'].iloc[count].split(" ; "))
    SG_entities = replace_spacestr_NA(df['StudyGroup'].iloc[count].split(" ; "))
    MO_entities = replace_spacestr_NA(df['Moderator'].iloc[count].split(" ; "))
    ST_entities = replace_spacestr_NA(df['StudyType'].iloc[count].split(" ; "))
    SE_entities = replace_spacestr_NA(df['StudyEnvironment'].iloc[count].split(" ; "))
    return AT_entities, HE_entities, ET_entities, SG_entities, MO_entities, ST_entities, SE_entities

def find_indices_words_in_sentence(entity_list, fullsentence):
    """Find the index of the phrases in the sentence."""
    indx = -1
    entity_indices = []
    for entity in entity_list:
        candidates = [(m.start()+1) for m in re.finditer(re.escape(" " + entity + " "), (" " + fullsentence + " ")) if m.start()>= indx][0]
        #matches = [(m.start()+1) for m in re.finditer(re.escape(" " + entity + " "), (" " + fullsentence + " ")) if m.start() >= indx]
        #if matches:
            #candidates = matches[0]
        entity_indices.extend([candidates])
        indx = max(entity_indices)
    return entity_indices


def repeat_each_item_n_times_in_list (list, times):
    """Repeat each item n times in a list"""
    rep_list = []
    for el in list:
        rep_list.extend([el] * times)
    return rep_list


def check_if_in_wordlist_and_append(list_of_lists, wordlist):
    """Checks if any element of a list in the list of lists is in the wordlist
       and then appends it to the output list."""
    output_list = []
    for list in list_of_lists:
        output_list.extend([el for el in list if el in wordlist])
    return output_list


def combinations(items):
    """Creates a list with all combinations of the elements in the items list. """
    comblist = []
    for i in range(0,len(items)):
        comblist.append(items[i])
        for x in range((i+2), (len(items)+1)):
            comblist.append(items[i:x])
            if (i+1) < x < len(items):
                comblist.append([items[i], items[x]])
                if x < (len(items)-1):
                    comblist.append(([items[i]] + items[x:]))
    return comblist

def combinations_NANlists(items):
    """Creates a list with all combinations of the elements in the items list,
       dealing with the issue of NaNs. """
    if items != ["NaN"]:
        if "NaN" in items: #one extra if-statement
            items.remove("NaN")
        comblist = combinations(items)
        comblist.extend(["NaN"])
        items.extend(["NaN"])
    else:
        comblist = items
    return comblist

def combinations_alllists(items):
    """Creates a nested list of lists of all combinations of
       the elements of the items list. """
    comblist = []
    for i in range(0,len(items)):
        comblist.append([items[i]])
        for x in range((i+2), (len(items)+1)):
            comblist.append(items[i:x])
            if (i+1) < x < len(items):
                comblist.append([items[i], items[x]])
                if x < (len(items)-1):
                    comblist.append(([items[i]] + items[x:]))
    return comblist


def evidence_instance_appending(ATs, HEs, ETs, SGs, MOs, STs, SEs): #2 additional labels so the repeats need to change
    """Creates lists for each label with the correct number of repeats of each phrase,
       so that when placed in a dataframe together it covers all combinations in the correct order."""
    AT_extension, HE_extension, ET_extension, SG_extension, MO_extension, SE_extension, ST_extension = [],[],[],[],[], [], []
    AT_combinations = combinations(ATs)
    SG_combinations = combinations_NANlists(SGs)
    ST_combinations = combinations_NANlists(STs)
    SE_combinations = combinations_NANlists(SEs)
    AT_extension.extend((AT_combinations) * len(HEs) * len(ETs) * len(SG_combinations) * len(ST_combinations) * len(SE_combinations) * len(MOs))
    HE_extension.extend((repeat_each_item_n_times_in_list(HEs, len(AT_combinations))) * len(ETs) * len(SG_combinations) * len(ST_combinations) * len(SE_combinations) * len(MOs))
    ET_extension.extend((repeat_each_item_n_times_in_list(ETs, (len(HEs)*len(AT_combinations)))) * len(SG_combinations) * len(ST_combinations) * len(SE_combinations) * len(MOs)) 
    SG_extension.extend((repeat_each_item_n_times_in_list((SG_combinations), (len(HEs)*len(ETs)*len(AT_combinations)))) * len(MOs)* len(ST_combinations) * len(SE_combinations) )
    ST_extension.extend((repeat_each_item_n_times_in_list(ST_combinations, (len(HEs)*len(ETs)*len(AT_combinations))))  * len(SE_combinations) *len(SG_combinations) * len(MOs)) #study-type
    SE_extension.extend((repeat_each_item_n_times_in_list(SE_combinations, (len(HEs)*len(ETs)*len(AT_combinations)))) * len(ST_combinations) *len(SG_combinations) * len(MOs)) #study-environemtn
    MO_extension.extend((repeat_each_item_n_times_in_list(MOs, (len(HEs)*len(ETs)*len(SG_combinations) * len(ST_combinations) * len(SE_combinations) * len(AT_combinations))))) 
    Nr_added = len(AT_combinations) * len(HEs) * len(ETs) * len(SG_combinations) * len(MOs) * len(ST_combinations) * len(SE_combinations)
    return AT_extension, HE_extension, ET_extension, SG_extension, MO_extension, ST_extension, SE_extension, Nr_added


def test_ET_joined_evidence_instance(ET_words, full_sentence, ET_indices): #exposure types
    """ Checks if multiple behavior options are mentioned in one evidence instance."""
    if len(ET_indices)>1:
        between_words = full_sentence[min(ET_indices):max(ET_indices)]
        for entity in ET_words:
            between_words = between_words.replace(entity, "")
        between_words = between_words.split()
        if all([True if word in ["and", ",", "the", "as", "well"] else False for word in between_words]):
            return 1
        else:
            return 0
    else:
        return 1


def check_if_varcombi_same_dependency_subtree(varlist_varcombi, totalsentence_varlist, subtrees):
    """ Checks whether the variable combination is part of the same dependency subtree."""
    joined_instance = 0
    missing_var = "NaN"
    remaining_var = [i for i in totalsentence_varlist if (i not in varlist_varcombi) and (i != "NaN")]
    variables = [x for x in varlist_varcombi if x != "NaN"]
    for a in range(0, len(subtrees)):
        if all([True if x in subtrees[a] else False for x in variables]):
            joined_instance += 1
            if any([True if x in subtrees[a] else False for x in remaining_var]):
                missing_var = "1"
            else:
                missing_var = "0"
    if joined_instance > 0:
        return "1", missing_var
    else:
        return "0", missing_var


def Dependencytree_to_Graph(doc):
    """ Load spacy's dependency tree into a networkx graph."""
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    return graph

#def getShortestPathbetweenPhrases(listA, listB, graph):
    """Find the minimum, maximum and sum of shortest path between two phrases
       (multiple words continuesly labeled with the same tag)."""
    #shortestPath = []
    #if not isinstance(listA, str):
        #words_inlistA = list(chain.from_iterable([phrase.split(" ") for phrase in listA]))
    #else:
        #words_inlistA = listA.split(" ")
    #if not isinstance(listB, str):
        #words_inlistB = list(chain.from_iterable([phrase.split(" ") for phrase in listB]))
    #else:
        #words_inlistB = listB.split(" ")
    #for f in range(0, len(words_inlistA)):
        #for x in range(0, len(words_inlistB)):
            #try:
                #shortestPath.extend([nx.shortest_path_length(graph, source=words_inlistA[f].lower(),
                                                             #target=words_inlistB[x].lower())])
            #except nx.exception.NodeNotFound:
                #print("Node not found", words_inlistA[f], words_inlistB[x])
            #except nx.exception.NetworkXNoPath:
                #print("No Path between", words_inlistA[f], "and", words_inlistB[x])
            #shortestPath.extend([100])
        #else:
            #pass
    #return [min(shortestPath)], [max(shortestPath)], [sum(shortestPath)]


import networkx as nx
from itertools import chain

def getShortestPathbetweenPhrases(listA, listB, graph):
    """Find the minimum, maximum, and sum of shortest path lengths between two phrases
       (multiple words continuously labeled with the same tag)."""
    shortestPath = []
    
    # Split phrases into individual words
    if not isinstance(listA, str):
        words_inlistA = list(chain.from_iterable([phrase.split(" ") for phrase in listA]))
    else:
        words_inlistA = listA.split(" ")
        
    if not isinstance(listB, str):
        words_inlistB = list(chain.from_iterable([phrase.split(" ") for phrase in listB]))
    else:
        words_inlistB = listB.split(" ")

    # Calculate shortest path lengths
    for f in range(len(words_inlistA)):
        for x in range(len(words_inlistB)):
            try:
                # Attempt to calculate the shortest path length
                path_length = nx.shortest_path_length(graph, source=words_inlistA[f].lower(), target=words_inlistB[x].lower())
                shortestPath.append(path_length)
            except nx.NodeNotFound:
                print(f"Node not found: {words_inlistA[f].lower()} or {words_inlistB[x].lower()} not in graph.")
            except nx.NetworkXNoPath:
                print(f"No path between nodes: {words_inlistA[f].lower()} and {words_inlistB[x].lower()}.")
                shortestPath.append(100)  # Use 100 as a placeholder for no path found

    # Handle case where no paths were found (empty shortestPath list)
    if not shortestPath:
        return [float('inf')], [0], [0]  # Placeholder values if no paths found

    return [min(shortestPath)], [max(shortestPath)], [sum(shortestPath)]


def print_summary_of_length_ofLists(list_oflists):
    """ Prints the length of all lists and names the outlier lists.
        Helps figuring out errors in case a target list was erroneous."""
    x = [len(element) for element in list_oflists]
    median = int(statistics.median(x))
    print("Median value: ", median, "appears", x.count(median), "times")
    outliers = [count for count, value in enumerate(x) if value != median]
    print("outliers are", outliers)

def ifnotlist_makelist(object):
    """ Makes a list out of non-lists."""
    if not isinstance(object, list):
        return [object]
    else:
        return object

def testVerbInBetween(verb_indx, idx1, idx2 ):
    """ Tests whether there is a verb in between idx1 and idx2. """
    result = ["1" if any(True for x in verb_indx if idx1 > x < idx2
                or idx1 < x > idx2) else "0"]
    return result

def testIfAllPhrasesBeforeIdx(idx, PhraseIndices):
    """ Tests whether all phrases in the variable combination are before a specific index
       or all are after. The important thing is whether they are together (either before or after)
       and not split. An example application is to test whether they are before or after
       split propositions or semicolons."""
    if idx != 'NaN':
        if all(True if x == 'NaN' or x > idx else False for x in PhraseIndices) or all(
                True if x == 'NaN' or x < idx else False for x in PhraseIndices):
            return ["1"]
        else:
            return ["0"]
    else:
        return ["NaN"]

# Execution

#########################################
### READING AND PREPARING DATA ##########
#########################################

os.chdir(r"C:/Users/4209416/OneDrive - Universiteit Utrecht/Desktop/ETAIN/PHIA_framework/Automated-KG/newCrossRef/Direct_Effects/final_kg")
#csv = os.path.join(os.getcwd(), ("Evidence_instances_df_ManualLabel.csv"))
csv = os.path.join(os.getcwd(), ("Evidence_instances_df.csv"))
#csv = os.path.join(os.getcwd(), ("Evidence_instances_df_training_data.csv"))
Evidence_instances_df = pd.read_csv(csv)
# columnnames: 'DOI','Sentence','ExposureType','HealthEffects','AssociationType', 'StudyType', 'StudyEnvironment',
#              'StudyGroup','Moderator',Fullsentence','ET_POS','HE_POS',AT_POS','sentence_POS'

## disaggregating evidence of within sentence colocation
complete_evidence_Instances = Evidence_instances_df.iloc[list(np.where((Evidence_instances_df['HealthEffects'].notnull()) & (Evidence_instances_df['AssociationType'].notnull()))[0])]
complete_evidence_Instances = complete_evidence_Instances.fillna(" ")

# Fix punctuation
complete_evidence_Instances = FixBracketsInVariableNames(complete_evidence_Instances)
complete_evidence_Instances['Fullsentence'] = [x.replace("?",",") for x in complete_evidence_Instances['Fullsentence']]

## subselecting complete sentences
complete_evidence_Instances = SelectGrammCompleteSent(complete_evidence_Instances)


#########################################################################################
### create df with all possible within sentence evidence relation combinations ##########
#########################################################################################
# the entity list has to be ordered sequentially;
# finds the indices of the words in the sentence

## Target List Creation
Evidence_Truth = []
HE_disagg, AT_disagg, ET_disagg, SG_disagg, MO_disagg, ST_disagg, SE_disagg,  = [], [], [], [], [], [], [] # twp additional labels
SENT_disagg, DOI_disagg, sent_ID_disagg = [], [], []

## Syntactical Properties
# Variable Combination Indices
HE_idx_disagg, AT_idx_disagg, ET_idx_disagg, MO_idx_disagg, SG_idx_disagg, ST_idx_disagg, SE_idx_disagg, split_prepos_idx_disagg = [], [], [], [], [], [], [], []
all_before_or_after_split_propos, index_dist_ATmin_HE, index_dist_ATmax_HE, index_dist_ATmin_ET, index_dist_ATmax_ET = [], [], [], [], []
index_dist_ATmin_MO, index_dist_ATmax_MO, index_dist_ATmin_SGmin, index_dist_ATmin_SGmax, index_dist_ATmax_SGmin, index_dist_ATmax_SGmax = [], [], [], [], [], []
index_dist_ATmin_STmin, index_dist_ATmin_STmax, index_dist_ATmax_STmin, index_dist_ATmax_STmax = [], [], [], [] #StudyType
index_dist_ATmin_SEmin, index_dist_ATmin_SEmax, index_dist_ATmax_SEmin, index_dist_ATmax_SEmax = [], [], [], [] #StudyEnvironment
semicolon_idx, all_before_or_after_semicolon = [], []
min_negator_idx, max_negator_idx = [], []

minimum_AT_idx, maximum_AT_idx, minimum_SG_idx, maximum_SG_idx = [], [], [], []
minimum_ST_idx, maximum_ST_idx = [], []
minimum_SE_idx, maximum_SE_idx = [], []

# sentence level indices
max_sent_ATidx, min_sent_ATidx, max_sent_HEidx, min_sent_HEidx, max_sent_ETidx, min_sent_ETidx = [], [], [], [], [], []
max_sent_MOidx, min_sent_MOidx, max_sent_SGidx, min_sent_SGidx= [], [], [], []
max_sent_STidx, min_sent_STidx = [], [] #StudyType
max_sent_SEidx, min_sent_SEidx = [], [] #StudyEnvironment

# Nr entities
Nr_AT, Nr_HE, Nr_ET, Nr_MO, Nr_SG, Nr_ST, Nr_SE  = [], [], [], [], [], [], []

# POS properties (part of speech)
Nr_verbs, earliestverb_indx, latestverb_indx = [], [], []

# Association Type properties
Verb_in_AT_inst, Noun_in_AT_inst, Adj_Adv_in_AT_inst, Comp_Adj_Adv_in_AT_inst, Verb_outside_AT_inst, Noun_outside_AT_inst, Adj_Adv_outside_AT_inst = [], [], [], [], [], [], []
Nr_ATs_in_instance, Multi_AT_Indice_gap = [], []
some_AT_in_brackets, all_AT_in_brackets= [], []
multiAT_shortestPathLen_min, multiAT_shortestPathLen_max = [], []
ATinst_has_not, ATsent_has_not = [], []

# combinatory variable properties
same_dependTree, missing_var_in_same_dependTree = [], []
AT_HE_minshortpath, AT_HE_maxshortpath, AT_HE_sumshortpath, AT_ET_minshortpath, AT_ET_maxshortpath, AT_ET_sumshortpath = [], [], [], [], [], []
AT_SG_minshortpath, AT_SG_maxshortpath, AT_SG_sumshortpath, AT_MO_minshortpath, AT_MO_maxshortpath, AT_MO_sumshortpath =  [], [], [], [], [], []
AT_ST_minshortpath, AT_ST_maxshortpath, AT_ST_sumshortpath = [], [], [] #StudyType
AT_SE_minshortpath, AT_SE_maxshortpath, AT_SE_sumshortpath = [], [], [] #StudyEnvironment
verb_between_ATmin_HE, verb_between_ATmax_HE, verb_between_ATmin_ET, verb_between_ATmax_ET = [], [], [], []
verb_between_ATmin_MO, verb_between_ATmax_MO, verb_between_ATmin_SGmin, verb_between_ATmax_SGmin, verb_between_ATmin_SGmax, verb_between_ATmax_SGmax = [], [], [], [], [], []
verb_between_ATmin_STmin, verb_between_ATmax_STmin, verb_between_ATmin_STmax, verb_between_ATmax_STmax = [], [], [], [] #StudyType
verb_between_ATmin_SEmin, verb_between_ATmax_SEmin, verb_between_ATmin_SEmax, verb_between_ATmax_SEmax = [], [], [], [] #StudyEnvironment


#others
joined_ET_instances = [] #exposure options

## filling target lists and syntactical properties
for count, value in enumerate(complete_evidence_Instances['Fullsentence']):
    ## creating entity and idx lists
    AT_entities, HE_entities, ET_entities, SG_entities, MO_entities, ST_entities, SE_entities = createEntityList(complete_evidence_Instances)

    AT_indices = find_indices_words_in_sentence(AT_entities, value)
    ET_indices = find_indices_words_in_sentence(ET_entities, value)
    HE_indices = find_indices_words_in_sentence(HE_entities, value)
    SG_indices = find_indices_words_in_sentence(SG_entities, value)
    MO_indices = find_indices_words_in_sentence(MO_entities, value)
    ST_indices = find_indices_words_in_sentence(ST_entities, value)
    SE_indices = find_indices_words_in_sentence(SE_entities, value)


    ## String based identification
    split_prepos = [m.start() for x in ["whereas", "while", "unlike", "although"] for m in re.finditer((x), value)]
    split_prepos.extend(["NaN"])
    ET_joined = test_ET_joined_evidence_instance(ET_words=ET_entities, full_sentence=value, ET_indices=ET_indices)
    # commas = [m.start() for m in re.finditer(",", value)]
    # brackets = [m.start() for x in ["(", ")"] for m in re.finditer((x), value)]
    semicolon = [m.start() for m in re.finditer(";", value)]
    semicolon.extend(["NaN"])
    negators = [m.start() for x in ["not", "except", "Not"] for m in re.finditer((x), value)]

    #######################
    ## Dependency Parser ##
    #######################
    doc = en(value)
    root = [word for word in doc if (word.dep_ in ['ROOT', 'advcl']) or ((word.head.dep_ == 'ROOT') and (word.dep_ == 'conj'))]
    rootpos = [token.pos_ for token in root]
    #print("ROOT", root, rootpos)

    ## for each root identify important subtree components and compile a wordlist of these
    ## then check which ETs, HEs, ATs, SGs, STs, SEs and MOs are part of the sublist
    items_in_subtree_groups = []
    for token in root:
        clausal_complement = [word for word in token.children if word.dep_ in ['ccomp', 'xcomp']]
        clausal_complement_children = [child for cl in clausal_complement for child in cl.subtree]
        subject = [child for child in token.children if child.dep_ in ['nsubjpass', 'nsubj']]
        subject_children = [child for sub in subject for child in sub.subtree]
        preposition = [child for child in token.children if child.dep_ == 'prep']
        preposition_children = [child for prep in preposition for child in prep.subtree]
        object = [child for child in token.children if child.dep_ == 'dobj']
        object_children = [child for obj in object for child in obj.subtree]
        wordlist = list(chain.from_iterable([[str(token)], clausal_complement, clausal_complement_children,
                                             subject, subject_children, preposition, preposition_children, object,
                                             object_children]))
        wordlist = (" ").join([str(word) for word in wordlist])
        items_in_subtree_groups.append(check_if_in_wordlist_and_append([AT_entities,HE_entities,ET_entities,SG_entities, ST_entities, SE_entities, MO_entities], wordlist))
    #print(items_in_subtree_groups)

    graph = Dependencytree_to_Graph(doc)

    ### filling target list
    # variable names
    ET_entities.extend(["NaN"])
    MO_entities.extend(["NaN"])
    SG_entities.extend(["NaN"])
    ST_entities.extend(["NaN"]) #StudyType
    SE_entities.extend(["NaN"]) #StudyEnvironment
    AT_extension_name, HE_extension_name, ET_extension_name, SG_extension_name, MO_extension_name, ST_extension_name, SE_extension_name, NR_poss_relat = evidence_instance_appending(AT_entities, HE_entities, ET_entities, SG_entities,MO_entities, ST_entities, SE_entities)
    HE_disagg.extend(HE_extension_name)
    AT_disagg.extend(AT_extension_name)
    ET_disagg.extend(ET_extension_name)    
    SG_disagg.extend(SG_extension_name)
    MO_disagg.extend(MO_extension_name)
    ST_disagg.extend(ST_extension_name) #StudyType
    SE_disagg.extend(SE_extension_name) #StudyEnvironment
   

    # indices
    ET_indices.extend(["NaN"])
    MO_indices.extend(["NaN"])
    SG_indices.extend(["NaN"])
    ST_indices.extend(["NaN"])
    SE_indices.extend(["NaN"])
    AT_extension_indx, HE_extension_indx, ET_extension_indx, SG_extension_indx, MO_extension_indx, ST_extension_indx, SE_extension_indx, NR_poss_relat  = evidence_instance_appending(AT_indices, HE_indices, ET_indices, SG_indices,MO_indices, ST_indices, SE_indices)
    HE_idx_disagg.extend(HE_extension_indx)
    AT_idx_disagg.extend(AT_extension_indx)
    ET_idx_disagg.extend(ET_extension_indx)
    MO_idx_disagg.extend(MO_extension_indx)
    SG_idx_disagg.extend(SG_extension_indx)
    ST_idx_disagg.extend(ST_extension_indx) #StudyType
    SE_idx_disagg.extend(SE_extension_indx) #StudyEnvironment

    ATindicelist = combinations_alllists(AT_indices)
    AT_repetition = int(NR_poss_relat/len(ATindicelist))
    Nr_ATs_in_instance.extend([len(x) for x in ATindicelist] * AT_repetition)
    minimum_AT_idx.extend([min(x) for x in ATindicelist] * AT_repetition)
    maximum_AT_idx.extend([max(x) for x in ATindicelist] * AT_repetition)
    Multi_AT_Indice_gap.extend([max(x)-min(x) for x in ATindicelist] * AT_repetition)

    # combinatory syntactical properties
    # part of same subtree + shortest path between all options
    for i in range(0, len(AT_extension_name)):
        same_tree, missing_var_sametree = check_if_varcombi_same_dependency_subtree(list(chain.from_iterable([ifnotlist_makelist(AT_extension_name[i]), ifnotlist_makelist(HE_extension_name[i]),
                                                  ifnotlist_makelist(ET_extension_name[i]), ifnotlist_makelist(SG_extension_name[i]), 
                                                  ifnotlist_makelist(ST_extension_name[i]), ifnotlist_makelist(SE_extension_name[i]), 
                                                  ifnotlist_makelist(MO_extension_name[i])
                                                  ])), 
                                                  list(chain.from_iterable([ET_entities, SG_entities, ST_entities, SE_entities,
                                                  MO_entities])), items_in_subtree_groups)
        same_dependTree.extend(same_tree)
        missing_var_in_same_dependTree.extend([missing_var_sametree])
        minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], HE_extension_name[i], graph)
        AT_HE_minshortpath.extend(minSP)
        AT_HE_maxshortpath.extend(maxSP)
        AT_HE_sumshortpath.extend(sumSP)
        if ET_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], ET_extension_name[i], graph)
            AT_ET_minshortpath.extend(minSP)
            AT_ET_maxshortpath.extend(maxSP)
            AT_ET_sumshortpath.extend(sumSP)
        else:
            AT_ET_minshortpath.extend(["NaN"])
            AT_ET_maxshortpath.extend(["NaN"])
            AT_ET_sumshortpath.extend(["NaN"])
        if SG_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], SG_extension_name[i], graph)
            AT_SG_minshortpath.extend(minSP)
            AT_SG_maxshortpath.extend(maxSP)
            AT_SG_sumshortpath.extend(sumSP)
        else:
            AT_SG_minshortpath.extend(["NaN"])
            AT_SG_maxshortpath.extend(["NaN"])
            AT_SG_sumshortpath.extend(["NaN"])
        if MO_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], MO_extension_name[i], graph)
            AT_MO_minshortpath.extend(minSP)
            AT_MO_maxshortpath.extend(maxSP)
            AT_MO_sumshortpath.extend(sumSP)
        else:
            AT_MO_minshortpath.extend(["NaN"])
            AT_MO_maxshortpath.extend(["NaN"])
            AT_MO_sumshortpath.extend(["NaN"])
        if ST_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], ST_extension_name[i], graph)
            AT_ST_minshortpath.extend(minSP)
            AT_ST_maxshortpath.extend(maxSP)
            AT_ST_sumshortpath.extend(sumSP)
        else:
            AT_ST_minshortpath.extend(["NaN"])
            AT_ST_maxshortpath.extend(["NaN"])
            AT_ST_sumshortpath.extend(["NaN"])
        if SE_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], SE_extension_name[i], graph)
            AT_SE_minshortpath.extend(minSP)
            AT_SE_maxshortpath.extend(maxSP)
            AT_SE_sumshortpath.extend(sumSP)
        else:
            AT_SE_minshortpath.extend(["NaN"])
            AT_SE_maxshortpath.extend(["NaN"])
            AT_SE_sumshortpath.extend(["NaN"])

    # POS properties
    sentencePOS = complete_evidence_Instances['sentence_POS'].iloc[count].split(" ; ")
    words_in_sentence = value.split(" ")
    verbs = [words_in_sentence[x+1] for x,v in enumerate(sentencePOS) if v in ["VBN", "VB", "VBZ", "VBP", "VBD"]]
    # verbsPOS = [v for v in sentencePOS if v in ["VBN", "VB", "VBZ", "VBP", "VBD"]]
    verb_indx = find_indices_words_in_sentence(verbs, value)
    Nr_verbs.extend([len(verbs)]*NR_poss_relat)
    if len(verbs) > 0:
        earliestverb_indx.extend([min(verb_indx)]*NR_poss_relat)
        latestverb_indx.extend([max(verb_indx)]*NR_poss_relat)
    else:
        earliestverb_indx.extend(["NaN"]*NR_poss_relat)
        latestverb_indx.extend(["NaN"]*NR_poss_relat)


    #split propos index
    AT_extension_indx_alllists = ATindicelist*AT_repetition
    for i in range(0, len(AT_extension_name)):
        indices = list(chain.from_iterable([AT_extension_indx_alllists[i], [HE_extension_indx[i]],[ET_extension_indx[i]], 
                                            ifnotlist_makelist(SG_extension_indx[i]), [MO_extension_indx[i]], 
                                            ifnotlist_makelist(ST_extension_indx[i]), [MO_extension_indx[i]], #StudyType
                                            ifnotlist_makelist(SE_extension_indx[i]), [MO_extension_indx[i]]])) #StudyEnvironment

        all_before_or_after_split_propos.extend(testIfAllPhrasesBeforeIdx(idx=split_prepos[0], PhraseIndices=indices))
        all_before_or_after_semicolon.extend(testIfAllPhrasesBeforeIdx(idx=semicolon[0], PhraseIndices=indices))

        index_dist_ATmin_HE.extend([min(AT_extension_indx_alllists[i]) - HE_extension_indx[i]])
        index_dist_ATmax_HE.extend([max(AT_extension_indx_alllists[i]) - HE_extension_indx[i]])

        verb_between_ATmin_HE.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), HE_extension_indx[i]))
        verb_between_ATmax_HE.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), HE_extension_indx[i]))
        if ET_extension_indx[i] != "NaN":
            index_dist_ATmin_ET.extend([min(AT_extension_indx_alllists[i]) - ET_extension_indx[i]])
            index_dist_ATmax_ET.extend([max(AT_extension_indx_alllists[i]) - ET_extension_indx[i]])

            verb_between_ATmin_ET.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), ET_extension_indx[i]))
            verb_between_ATmax_ET.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), ET_extension_indx[i]))
        else:
            index_dist_ATmin_ET.extend(["NaN"])
            index_dist_ATmax_ET.extend(["NaN"])
            verb_between_ATmin_ET.extend(["NaN"])
            verb_between_ATmax_ET.extend(["NaN"])
        if MO_extension_indx[i] != "NaN":
            index_dist_ATmin_MO.extend([min(AT_extension_indx_alllists[i]) - MO_extension_indx[i]])
            index_dist_ATmax_MO.extend([max(AT_extension_indx_alllists[i]) - MO_extension_indx[i]])

            verb_between_ATmin_MO.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), MO_extension_indx[i]))
            verb_between_ATmax_MO.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), MO_extension_indx[i]))
        else:
            index_dist_ATmin_MO.extend(["NaN"])
            index_dist_ATmax_MO.extend(["NaN"])
            verb_between_ATmin_MO.extend(["NaN"])
            verb_between_ATmax_MO.extend(["NaN"])
        if SG_extension_indx[i] != "NaN":
            index_dist_ATmin_SGmin.extend([min(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_ATmax_SGmin.extend([max(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_ATmin_SGmax.extend([min(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_ATmax_SGmax.extend([max(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(SG_extension_indx[i]))])

            verb_between_ATmin_SGmin.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_ATmax_SGmin.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_ATmin_SGmax.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_ATmax_SGmax.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(SG_extension_indx[i]))))

            minimum_SG_idx.extend([min(ifnotlist_makelist(SG_extension_indx[i]))])
            maximum_SG_idx.extend([max(ifnotlist_makelist(SG_extension_indx[i]))])
        else:
            index_dist_ATmin_SGmin.extend(["NaN"])
            index_dist_ATmax_SGmin.extend(["NaN"])
            index_dist_ATmin_SGmax.extend(["NaN"])
            index_dist_ATmax_SGmax.extend(["NaN"])
            verb_between_ATmin_SGmin.extend(["NaN"])
            verb_between_ATmax_SGmin.extend(["NaN"])
            verb_between_ATmin_SGmax.extend(["NaN"])
            verb_between_ATmax_SGmax.extend(["NaN"])
            minimum_SG_idx.extend(["NaN"])
            maximum_SG_idx.extend(["NaN"])
        if ST_extension_indx[i] != "NaN": #StudyType
            index_dist_ATmin_STmin.extend([min(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(ST_extension_indx[i]))])
            index_dist_ATmax_STmin.extend([max(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(ST_extension_indx[i]))])
            index_dist_ATmin_STmax.extend([min(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(ST_extension_indx[i]))])
            index_dist_ATmax_STmax.extend([max(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(ST_extension_indx[i]))])

            verb_between_ATmin_STmin.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(ST_extension_indx[i]))))
            verb_between_ATmax_STmin.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(ST_extension_indx[i]))))
            verb_between_ATmin_STmax.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(ST_extension_indx[i]))))
            verb_between_ATmax_STmax.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(ST_extension_indx[i]))))

            minimum_ST_idx.extend([min(ifnotlist_makelist(ST_extension_indx[i]))])
            maximum_ST_idx.extend([max(ifnotlist_makelist(ST_extension_indx[i]))])
        else:
            index_dist_ATmin_STmin.extend(["NaN"])
            index_dist_ATmax_STmin.extend(["NaN"])
            index_dist_ATmin_STmax.extend(["NaN"])
            index_dist_ATmax_STmax.extend(["NaN"])
            verb_between_ATmin_STmin.extend(["NaN"])
            verb_between_ATmax_STmin.extend(["NaN"])
            verb_between_ATmin_STmax.extend(["NaN"])
            verb_between_ATmax_STmax.extend(["NaN"])
            minimum_ST_idx.extend(["NaN"])
            maximum_ST_idx.extend(["NaN"])
        if SE_extension_indx[i] != "NaN": #StudyEnvironment
            index_dist_ATmin_SEmin.extend([min(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(SE_extension_indx[i]))])
            index_dist_ATmax_SEmin.extend([max(AT_extension_indx_alllists[i]) - min(ifnotlist_makelist(SE_extension_indx[i]))])
            index_dist_ATmin_SEmax.extend([min(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(SE_extension_indx[i]))])
            index_dist_ATmax_SEmax.extend([max(AT_extension_indx_alllists[i]) - max(ifnotlist_makelist(SE_extension_indx[i]))])

            verb_between_ATmin_SEmin.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(SE_extension_indx[i]))))
            verb_between_ATmax_SEmin.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), min(ifnotlist_makelist(SE_extension_indx[i]))))
            verb_between_ATmin_SEmax.extend(testVerbInBetween(verb_indx, min(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(SE_extension_indx[i]))))
            verb_between_ATmax_SEmax.extend(testVerbInBetween(verb_indx, max(AT_extension_indx_alllists[i]), max(ifnotlist_makelist(SE_extension_indx[i]))))

            minimum_SE_idx.extend([min(ifnotlist_makelist(SE_extension_indx[i]))])
            maximum_SE_idx.extend([max(ifnotlist_makelist(SE_extension_indx[i]))])
        else:
            index_dist_ATmin_SEmin.extend(["NaN"])
            index_dist_ATmax_SEmin.extend(["NaN"])
            index_dist_ATmin_SEmax.extend(["NaN"])
            index_dist_ATmax_SEmax.extend(["NaN"])
            verb_between_ATmin_SEmin.extend(["NaN"])
            verb_between_ATmax_SEmin.extend(["NaN"])
            verb_between_ATmin_SEmax.extend(["NaN"])
            verb_between_ATmax_SEmax.extend(["NaN"])
            minimum_SE_idx.extend(["NaN"])
            maximum_SE_idx.extend(["NaN"])


    # Association types properties
    # if it contains a negator (not)
    ATsent_has_not.extend([1 if any(True if "not" in phrase else False for phrase in AT_entities) else 0] * NR_poss_relat)

    # direct link between association types
    ATentitylist = combinations_alllists(AT_entities)
    # Get the length and path
    multiAT_shortestPath_min, multiAT_shortestPath_max = [], []
    has_not = []
    for i in range(0, len(ATentitylist)):
        if len(ATentitylist[i]) == 1:
            multiAT_shortestPath_min.extend([0])
            multiAT_shortestPath_max.extend([0])
            has_not.extend([1 if "not" in ATentitylist[i] else 0])
        else:
            multiAT_shortestPath = []
            words_inlist = list(chain.from_iterable([phrase.split(" ") for phrase in ATentitylist[i]]))
            ## checking if any "not" in wordlist
            has_not.extend([1 if "not" in words_inlist else 0])
            for f in range(0, len(words_inlist)):
                for x in range((f + 1), (len(words_inlist))):
                    multiAT_shortestPath.extend([nx.shortest_path_length(graph, source=words_inlist[f].lower(),
                                                                   target=words_inlist[x].lower())])

            multiAT_shortestPath_min.extend([min(multiAT_shortestPath)])
            multiAT_shortestPath_max.extend([max(multiAT_shortestPath)])
    multiAT_shortestPathLen_min.extend(multiAT_shortestPath_min * AT_repetition)
    multiAT_shortestPathLen_max.extend(multiAT_shortestPath_max * AT_repetition)
    ATinst_has_not.extend(has_not * AT_repetition)

    # checking ATs part of speech (gramma function)
    fullsentence_ATPOS = complete_evidence_Instances['AT_POS'].iloc[count].split(" ; ")
    wordlengths = [len(word.split(" ")) for word in AT_entities]
    ATPOS_perword = [fullsentence_ATPOS[:wordlengths[0]]]
    for x in range(1,len(wordlengths)):
        idx = (sum(wordlengths[:x]))
        ATPOS_perword.extend([fullsentence_ATPOS[idx:idx+wordlengths[x]]])
    AT_POS_combi = combinations_alllists(ATPOS_perword)
    verb_inside = ["1" if any([True for i in ["VBN", "VB", "VBZ", "VBP", "VBD"] if i in list(chain.from_iterable(ATPOS))]) else "0" for ATPOS in AT_POS_combi]
    noun_inside = ["1" if any([True for i in ["NNS","NN"] if i in list(chain.from_iterable(ATPOS))]) else "0" for ATPOS in AT_POS_combi]
    adj_adv_inside = ["1" if any([True for i in ["JJ","RB", "JJS", "RBS"] if i in list(chain.from_iterable(ATPOS))]) else "0" for ATPOS in AT_POS_combi]
    comp_adj_adv_inside = ["1" if any([True for i in ["JJR","RBR"] if i in list(chain.from_iterable(ATPOS))]) else "0" for ATPOS in AT_POS_combi]
    Verb_in_AT_inst.extend((verb_inside) * AT_repetition)
    Noun_in_AT_inst.extend((noun_inside)* AT_repetition)
    Adj_Adv_in_AT_inst.extend((adj_adv_inside)*AT_repetition)
    Comp_Adj_Adv_in_AT_inst.extend((comp_adj_adv_inside)* AT_repetition)
    # Adj_Adv_outside_AT_inst
    # Verb_outside_AT_inst
    # Noun_outside_AT_inst
    # if it is before or after

    # some_AT_in_brackets
    # all_AT_in_brackets


    # sentence level variables
    split_prepos_idx_disagg.extend([split_prepos[0]] * NR_poss_relat)
    semicolon_idx.extend([semicolon[0]]*NR_poss_relat)
    if negators:
        min_negator_idx.extend([min(negators)]*NR_poss_relat)
        max_negator_idx.extend([max(negators)]*NR_poss_relat)
    else:
        min_negator_idx.extend(["NaN"] * NR_poss_relat)
        max_negator_idx.extend(["NaN"] * NR_poss_relat)
    SENT_disagg.extend([value] * NR_poss_relat)
    DOI_disagg.extend([complete_evidence_Instances['DOI'].iloc[count]] * NR_poss_relat)
    sent_ID_disagg.extend([complete_evidence_Instances['Sentence'].iloc[count]] * NR_poss_relat)
    Nr_AT.extend([len(AT_entities)]* NR_poss_relat)
    Nr_ET.extend([len(ET_entities)-1]* NR_poss_relat)
    Nr_HE.extend([len(HE_entities)]* NR_poss_relat)
    Nr_MO.extend([len(MO_entities)-1]* NR_poss_relat)
    Nr_SG.extend([len(SG_entities)-1]* NR_poss_relat)
    Nr_ST.extend([len(ST_entities)-1]* NR_poss_relat) #StudyType
    Nr_SE.extend([len(SE_entities)-1]* NR_poss_relat) #StudyEnvironment
    joined_ET_instances.extend([ET_joined]* NR_poss_relat)
    Evidence_Truth.extend(["0"]*NR_poss_relat)
    max_sent_ATidx.extend([max(AT_indices)] * NR_poss_relat)
    min_sent_ATidx.extend([min(AT_indices)] * NR_poss_relat)
    max_sent_HEidx.extend([max(HE_indices)] * NR_poss_relat)
    min_sent_HEidx.extend([min(HE_indices)] * NR_poss_relat)
    if ET_indices != ["NaN"]:
        ET_indices.remove("NaN")
        max_sent_ETidx.extend([max(ET_indices)] * NR_poss_relat)
        min_sent_ETidx.extend([min(ET_indices)] * NR_poss_relat)
    else:
        max_sent_ETidx.extend(["NaN"]*NR_poss_relat)
        min_sent_ETidx.extend(["NaN"]*NR_poss_relat)
    if MO_indices != ["NaN"]:
        MO_indices.remove("NaN")
        max_sent_MOidx.extend([max(MO_indices)] * NR_poss_relat)
        min_sent_MOidx.extend([min(MO_indices)] * NR_poss_relat)
    else:
        max_sent_MOidx.extend(["NaN"]*NR_poss_relat)
        min_sent_MOidx.extend(["NaN"]*NR_poss_relat)
    if SG_indices != ["NaN"]:
        SG_indices.remove("NaN")
        max_sent_SGidx.extend([max(SG_indices)] * NR_poss_relat)
        min_sent_SGidx.extend([min(SG_indices)] * NR_poss_relat)
    else:
        max_sent_SGidx.extend(["NaN"]*NR_poss_relat)
        min_sent_SGidx.extend(["NaN"]*NR_poss_relat)
    if ST_indices != ["NaN"]:
        ST_indices.remove("NaN")
        max_sent_STidx.extend([max(ST_indices)] * NR_poss_relat)
        min_sent_STidx.extend([min(ST_indices)] * NR_poss_relat)
    else:
        max_sent_STidx.extend(["NaN"]*NR_poss_relat)
        min_sent_STidx.extend(["NaN"]*NR_poss_relat)
    if SE_indices != ["NaN"]:
        SE_indices.remove("NaN")
        max_sent_SEidx.extend([max(SE_indices)] * NR_poss_relat)
        min_sent_SEidx.extend([min(SE_indices)] * NR_poss_relat)
    else:
        max_sent_SEidx.extend(["NaN"]*NR_poss_relat)
        min_sent_SEidx.extend(["NaN"]*NR_poss_relat)

    # check if all lists same length
    print_summary_of_length_ofLists([HE_disagg, AT_disagg, ET_disagg, SG_disagg, ST_disagg, SE_disagg, MO_disagg, SENT_disagg, DOI_disagg, sent_ID_disagg,
                                     HE_idx_disagg, AT_idx_disagg, ET_idx_disagg, MO_idx_disagg, SG_idx_disagg, ST_idx_disagg, SE_idx_disagg, 
                                     split_prepos_idx_disagg,Nr_AT, Nr_ET, Nr_HE, Nr_MO, Nr_SG, Nr_ST, Nr_SE, Nr_verbs, earliestverb_indx,
                                     latestverb_indx, Verb_in_AT_inst, Noun_in_AT_inst, Adj_Adv_in_AT_inst, Comp_Adj_Adv_in_AT_inst,
                                     Nr_ATs_in_instance, Multi_AT_Indice_gap, minimum_AT_idx, maximum_AT_idx, multiAT_shortestPathLen_min,
                                     multiAT_shortestPathLen_max, same_dependTree, missing_var_in_same_dependTree, joined_ET_instances,
                                     ATinst_has_not, ATsent_has_not, AT_HE_minshortpath, AT_HE_maxshortpath,
                                     AT_HE_sumshortpath, AT_ET_minshortpath, AT_ET_maxshortpath, AT_ET_sumshortpath, AT_SG_minshortpath,
                                     AT_SG_maxshortpath, AT_SG_sumshortpath, AT_MO_minshortpath, AT_MO_maxshortpath, AT_MO_sumshortpath,
                                     AT_ST_minshortpath, AT_ST_maxshortpath, AT_ST_sumshortpath,
                                     AT_SE_minshortpath, AT_SE_maxshortpath, AT_SE_sumshortpath,
                                     all_before_or_after_split_propos, index_dist_ATmin_HE, index_dist_ATmax_HE, index_dist_ATmin_ET,
                                     index_dist_ATmax_ET, index_dist_ATmin_MO, index_dist_ATmax_MO, index_dist_ATmin_SGmin, index_dist_ATmin_SGmax,
                                     index_dist_ATmax_SGmin, index_dist_ATmax_SGmax,
                                     index_dist_ATmin_STmin, index_dist_ATmin_STmax, index_dist_ATmax_STmin, index_dist_ATmax_STmax,
                                     index_dist_ATmin_SEmin, index_dist_ATmin_SEmax, index_dist_ATmax_SEmin, index_dist_ATmax_SEmax,
                                     verb_between_ATmin_HE, verb_between_ATmax_HE, verb_between_ATmin_ET, verb_between_ATmax_ET, verb_between_ATmin_MO,
                                     verb_between_ATmax_MO, verb_between_ATmin_SGmin, verb_between_ATmax_SGmin, verb_between_ATmin_SGmax, verb_between_ATmax_SGmax,
                                     verb_between_ATmin_STmin, verb_between_ATmax_STmin, verb_between_ATmin_STmax, verb_between_ATmax_STmax,
                                     verb_between_ATmin_SEmin, verb_between_ATmax_SEmin, verb_between_ATmin_SEmax, verb_between_ATmax_SEmax,
                                     max_sent_ATidx, min_sent_ATidx, max_sent_SGidx, min_sent_SGidx,
                                     max_sent_HEidx, min_sent_HEidx, max_sent_ETidx, min_sent_ETidx, max_sent_MOidx, min_sent_MOidx,
                                     max_sent_SGidx, min_sent_SGidx, max_sent_STidx, min_sent_STidx, max_sent_SEidx, min_sent_SEidx,
                                     semicolon_idx, all_before_or_after_semicolon, max_negator_idx, min_negator_idx])



possible_evidence_instances = pd.DataFrame({'DOI': DOI_disagg, 'Sentence': sent_ID_disagg, 'Evidence_Truth': Evidence_Truth, 'Fullsentence': SENT_disagg,
                                                 'ExposureType': ET_disagg, 'HealthEffects': HE_disagg,
                                                 'AssociationType': AT_disagg, 'StudyGroup': SG_disagg, 'StudyType': ST_disagg, 'StudyEnvironment': SE_disagg, 'Moderator': MO_disagg,
                                                 'HE_idx': HE_idx_disagg, 'ET_idx': ET_idx_disagg,
                                                 'MO_idx': MO_idx_disagg, 'Split_Propos_idx': split_prepos_idx_disagg,
                                                 'all_before_or_after_split_propos': all_before_or_after_split_propos,
                                                 'semicolon_idx': semicolon_idx, 'all_before_or_after_semicolon': all_before_or_after_semicolon,
                                                 'max_negator_idx': max_negator_idx, 'min_negator_idx': min_negator_idx,
                                                 'index_dist_ATmin_HE': index_dist_ATmin_HE, 'index_dist_ATmax_HE': index_dist_ATmax_HE,
                                                 'index_dist_ATmin_ET': index_dist_ATmin_ET, 'index_dist_ATmax_ET': index_dist_ATmax_ET,
                                                 'index_dist_ATmin_MO': index_dist_ATmin_MO, 'index_dist_ATmax_MO': index_dist_ATmax_MO,
                                                 'index_dist_ATmin_SGmin': index_dist_ATmin_SGmin,  'index_dist_ATmax_SGmin': index_dist_ATmax_SGmin,
                                                 'index_dist_ATmin_SGmax': index_dist_ATmin_SGmax, 'index_dist_ATmax_SGmax': index_dist_ATmax_SGmax,
                                                 'index_dist_ATmin_STmin': index_dist_ATmin_STmin,  'index_dist_ATmax_STmin': index_dist_ATmax_STmin,
                                                 'index_dist_ATmin_STmax': index_dist_ATmin_STmax, 'index_dist_ATmax_STmax': index_dist_ATmax_STmax,
                                                 'index_dist_ATmin_SEmin': index_dist_ATmin_SEmin,  'index_dist_ATmax_SEmin': index_dist_ATmax_SEmin,
                                                 'index_dist_ATmin_SEmax': index_dist_ATmin_SEmax, 'index_dist_ATmax_SEmax': index_dist_ATmax_SEmax,
                                                 'same_dependTree': same_dependTree, 'missing_var_in_same_dependTree' : missing_var_in_same_dependTree,  'joined_ET_instances': joined_ET_instances,
                                                 'Nr_AT': Nr_AT, 'Nr_ET': Nr_ET, 'Nr_HE': Nr_HE, 'Nr_MO': Nr_MO, 'Nr_SG': Nr_SG, 'Nr_ST': Nr_ST, 'Nr_SE': Nr_SE,
                                                 'Verb_in_AT_inst': Verb_in_AT_inst, 'Noun_in_AT_inst': Noun_in_AT_inst,
                                                 'Adj_Adv_in_AT_inst': Adj_Adv_in_AT_inst, 'Comp_Adj_Adv_in_AT_inst': Comp_Adj_Adv_in_AT_inst,
                                                 'Nr_ATs_in_instance': Nr_ATs_in_instance, 'Multi_AT_Indice_gap': Multi_AT_Indice_gap,
                                                 'minimum_AT_idx': minimum_AT_idx, 'maximum_AT_idx': maximum_AT_idx,
                                                 'minimum_SG_idx': minimum_SG_idx, 'maximum_SG_idx': maximum_SG_idx,
                                                 'minimum_ST_idx': minimum_ST_idx, 'maximum_ST_idx': maximum_ST_idx,
                                                 'minimum_SE_idx': minimum_SE_idx, 'maximum_SE_idx': maximum_SE_idx,
                                                 'ATinst_has_not': ATinst_has_not, 'ATsent_has_not': ATsent_has_not,
                                                 'Nr_verbs': Nr_verbs, 'earliestverb_indx': earliestverb_indx, 'latestverb_indx': latestverb_indx,
                                                 'multiAT_shortestPathLen_min': multiAT_shortestPathLen_min,'multiAT_shortestPathLen_max': multiAT_shortestPathLen_max,
                                                 'AT_HE_minshortpath': AT_HE_minshortpath, 'AT_HE_maxshortpath': AT_HE_maxshortpath, 'AT_HE_sumshortpath': AT_HE_sumshortpath,
                                                 'AT_ET_minshortpath': AT_ET_minshortpath, 'AT_ET_maxshortpath': AT_ET_maxshortpath, 'AT_ET_sumshortpath': AT_ET_sumshortpath,
                                                 'AT_SG_minshortpath': AT_SG_minshortpath, 'AT_SG_maxshortpath': AT_SG_maxshortpath, 'AT_SG_sumshortpath': AT_SG_sumshortpath,
                                                 'AT_ST_minshortpath': AT_ST_minshortpath, 'AT_ST_maxshortpath': AT_ST_maxshortpath, 'AT_ST_sumshortpath': AT_ST_sumshortpath,
                                                 'AT_SE_minshortpath': AT_SE_minshortpath, 'AT_SE_maxshortpath': AT_SE_maxshortpath, 'AT_SE_sumshortpath': AT_SE_sumshortpath,
                                                 'AT_MO_minshortpath': AT_MO_minshortpath, 'AT_MO_maxshortpath': AT_MO_maxshortpath, 'AT_MO_sumshortpath': AT_MO_sumshortpath,
                                                 'verb_between_ATmin_HE': verb_between_ATmin_HE, 'verb_between_ATmax_HE': verb_between_ATmax_HE,
                                                 'verb_between_ATmin_ET': verb_between_ATmin_ET, 'verb_between_ATmax_ET': verb_between_ATmax_ET,
                                                 'verb_between_ATmin_MO': verb_between_ATmin_MO, 'verb_between_ATmax_MO': verb_between_ATmax_MO,
                                                 'verb_between_ATmin_SGmin': verb_between_ATmin_SGmin, 'verb_between_ATmax_SGmin': verb_between_ATmax_SGmin,
                                                 'verb_between_ATmin_SGmax': verb_between_ATmin_SGmax, 'verb_between_ATmax_SGmax': verb_between_ATmax_SGmax,
                                                 'verb_between_ATmin_STmin': verb_between_ATmin_STmin, 'verb_between_ATmax_STmin': verb_between_ATmax_STmin,
                                                 'verb_between_ATmin_STmax': verb_between_ATmin_STmax, 'verb_between_ATmax_STmax': verb_between_ATmax_STmax,
                                                 'verb_between_ATmin_SEmin': verb_between_ATmin_SEmin, 'verb_between_ATmax_SEmin': verb_between_ATmax_SEmin,
                                                 'verb_between_ATmin_SEmax': verb_between_ATmin_SEmax, 'verb_between_ATmax_SEmax': verb_between_ATmax_SEmax,
                                                 'max_sent_ATidx': max_sent_ATidx, 'min_sent_ATidx': min_sent_ATidx,
                                                 'max_sent_HEidx': max_sent_HEidx, 'min_sent_HEidx': min_sent_HEidx,
                                                 'max_sent_ETidx': max_sent_ETidx, 'min_sent_ETidx': min_sent_ETidx,
                                                 'max_sent_MOidx': max_sent_MOidx, 'min_sent_MOidx': min_sent_MOidx,
                                                 'max_sent_SGidx': max_sent_SGidx, 'min_sent_SGidx': min_sent_SGidx,
                                                 'max_sent_STidx': max_sent_STidx, 'min_sent_SGidx': min_sent_STidx,
                                                 'max_sent_SEidx': max_sent_SEidx, 'min_sent_SGidx': min_sent_SEidx})

possible_evidence_instances.replace({'NaN': -100}, regex=False, inplace=True)

possible_evidence_instances.to_csv("possible_evidence_instances4.csv", index=False)
#possible_evidence_instances.to_csv("possible_evidence_instances_Manual4.csv", index=False)
# #
print("Nr of unique articles contributing evidence: ",len(set(possible_evidence_instances['DOI'])))