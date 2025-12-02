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
    df['Relationship'] = [x.replace("; ( ", "; ").replace(" ) ;", " ;").replace(" ) ", " ; ").replace(" ( ", " ; ").replace("( ","").replace(
            " )", "").replace("(", "").replace(")", "").replace("  ", " ") for x in df['Relationship']]
    df['IndirectEffects'] = [x.replace("; ( ", "; ").replace(" ) ;", " ;").replace(" ) ", " ; ").replace(" ( ", " ; ").replace("( ","").replace(
            " )", "").replace("(", "").replace(")", "").replace("  ", " ").replace(" +", "") for x in df['IndirectEffects']]
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
    RE_entities = df['Relationship'].iloc[count].split(" ; ")
    IE_entities = df['IndirectEffects'].iloc[count].split(" ; ")
    DE_entities = replace_spacestr_NA(df['DirectEffects'].iloc[count].split(" ; "))
    SG_entities = replace_spacestr_NA(df['StudyGroup'].iloc[count].split(" ; "))
    EC_entities = replace_spacestr_NA(df['EcoConsequences'].iloc[count].split(" ; "))
    return RE_entities, IE_entities, DE_entities, SG_entities, EC_entities

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


def evidence_instance_appending(REs, IEs, DEs, SGs, ECs): 
    """Creates lists for each label with the correct number of repeats of each phrase,
       so that when placed in a dataframe together it covers all combinations in the correct order."""
    RE_extension, IE_extension, DE_extension, SG_extension, EC_extension = [],[],[],[],[]
    RE_combinations = combinations(REs)
    SG_combinations = combinations_NANlists(SGs)
    #ST_combinations = combinations_NANlists(STs)
    #SE_combinations = combinations_NANlists(SEs)
    RE_extension.extend((RE_combinations) * len(IEs) * len(DEs) * len(SG_combinations) * len(ECs))
    IE_extension.extend((repeat_each_item_n_times_in_list(IEs, len(RE_combinations))) * len(DEs) * len(SG_combinations) * len(ECs))
    DE_extension.extend((repeat_each_item_n_times_in_list(DEs, (len(IEs)*len(RE_combinations)))) * len(SG_combinations) * len(ECs)) 
    SG_extension.extend((repeat_each_item_n_times_in_list((SG_combinations), (len(IEs)*len(DEs)*len(RE_combinations)))) * len(ECs))
    EC_extension.extend((repeat_each_item_n_times_in_list(ECs, (len(IEs)*len(DEs)*len(SG_combinations) * len(RE_combinations))))) 
    Nr_added = len(RE_combinations) * len(IEs) * len(DEs) * len(SG_combinations) * len(ECs)
    return RE_extension, IE_extension, DE_extension, SG_extension, EC_extension, Nr_added


def test_DE_joined_evidence_instance(DE_words, full_sentence, DE_indices): #behavior options
    """ Checks if multiple behavior options are mentioned in one evidence instance."""
    if len(DE_indices)>1:
        between_words = full_sentence[min(DE_indices):max(DE_indices)]
        for entity in DE_words:
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

os.chdir(r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects\final_kg")
#csv = os.path.join(os.getcwd(), ("Evidence_instances_df_ManualLabel.csv"))
csv = os.path.join(os.getcwd(), ("Evidence_instances_df.csv"))
#csv = os.path.join(os.getcwd(), ("Evidence_instances_df_training_data.csv"))
Evidence_instances_df = pd.read_csv(csv)
# columnnames: 'DOI','Sentence','I-DirectEffects','I-IndirectEffects','RelationshipType','I-EcoConsequences',
#              'StudyGroup','Moderator',Fullsentence','DE_POS','IE_POS',RT_POS','sentence_POS'

## disaggregating evidence of within sentence colocation
complete_evidence_Instances = Evidence_instances_df.iloc[list(np.where((Evidence_instances_df['IndirectEffects'].notnull()) & (Evidence_instances_df['Relationship'].notnull()))[0])]
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
IE_disagg, RE_disagg, DE_disagg, SG_disagg, EC_disagg  = [], [], [], [], [] 
SENT_disagg, DOI_disagg, sent_ID_disagg = [], [], []

## Syntactical Properties
# Variable Combination Indices
IE_idx_disagg, RE_idx_disagg, DE_idx_disagg, EC_idx_disagg, SG_idx_disagg, split_prepos_idx_disagg = [], [], [], [], [], []
all_before_or_after_split_propos, index_dist_REmin_IE, index_dist_REmax_IE, index_dist_REmin_DE, index_dist_REmax_DE = [], [], [], [], []
index_dist_REmin_EC, index_dist_REmax_EC, index_dist_REmin_SGmin, index_dist_REmin_SGmax, index_dist_REmax_SGmin, index_dist_REmax_SGmax = [], [], [], [], [], []
semicolon_idx, all_before_or_after_semicolon = [], []
min_negator_idx, max_negator_idx = [], []

minimum_RE_idx, maximum_RE_idx, minimum_SG_idx, maximum_SG_idx = [], [], [], []

# sentence level indices
max_sent_REidx, min_sent_REidx, max_sent_IEidx, min_sent_IEidx, max_sent_DEidx, min_sent_DEidx = [], [], [], [], [], []
max_sent_ECidx, min_sent_ECidx, max_sent_SGidx, min_sent_SGidx= [], [], [], []

# Nr entities
Nr_RE, Nr_IE, Nr_DE, Nr_EC, Nr_SG = [], [], [], [], []

# POS properties (part of speech)
Nr_verbs, earliestverb_indx, latestverb_indx = [], [], []

# Relationship Type properties
Verb_in_RE_inst, Noun_in_RE_inst, Adj_Adv_in_RE_inst, Comp_Adj_Adv_in_RE_inst, Verb_outside_RE_inst, Noun_outside_RE_inst, Adj_Adv_outside_RE_inst = [], [], [], [], [], [], []
Nr_REs_in_instance, Multi_RE_Indice_gap = [], []
some_RE_in_brackets, all_RE_in_brackets= [], []
multiRE_shortestPathLen_min, multiRE_shortestPathLen_max = [], []
REinst_has_not, REsent_has_not = [], []

# combinatory variable properties
same_dependTree, missing_var_in_same_dependTree = [], []
RE_IE_minshortpath, RE_IE_maxshortpath, RE_IE_sumshortpath, RE_DE_minshortpath, RE_DE_maxshortpath, RE_DE_sumshortpath = [], [], [], [], [], []
RE_SG_minshortpath, RE_SG_maxshortpath, RE_SG_sumshortpath, RE_EC_minshortpath, RE_EC_maxshortpath, RE_EC_sumshortpath =  [], [], [], [], [], []
verb_between_REmin_IE, verb_between_REmax_IE, verb_between_REmin_DE, verb_between_REmax_DE = [], [], [], []
verb_between_REmin_EC, verb_between_REmax_EC, verb_between_REmin_SGmin, verb_between_REmax_SGmin, verb_between_REmin_SGmax, verb_between_REmax_SGmax = [], [], [], [], [], []

#others
joined_DE_instances = [] #direct effects

## filling target lists and syntactical properties
for count, value in enumerate(complete_evidence_Instances['Fullsentence']):
    ## creating entity and idx lists
    RE_entities, IE_entities, DE_entities, SG_entities, EC_entities = createEntityList(complete_evidence_Instances)

    RE_indices = find_indices_words_in_sentence(RE_entities, value)
    DE_indices = find_indices_words_in_sentence(DE_entities, value)
    IE_indices = find_indices_words_in_sentence(IE_entities, value)
    SG_indices = find_indices_words_in_sentence(SG_entities, value)
    EC_indices = find_indices_words_in_sentence(EC_entities, value)
    

    ## String based identification
    split_prepos = [m.start() for x in ["whereas", "while", "unlike", "although"] for m in re.finditer((x), value)]
    split_prepos.extend(["NaN"])
    DE_joined = test_DE_joined_evidence_instance(DE_words=DE_entities, full_sentence=value, DE_indices=DE_indices)
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
    ## then check which BOs, BDs, ATs, SGs and MOs are part of the sublist
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
        items_in_subtree_groups.append(check_if_in_wordlist_and_append([RE_entities,IE_entities,DE_entities,SG_entities, EC_entities], wordlist))
    #print(items_in_subtree_groups)

    graph = Dependencytree_to_Graph(doc)

    ### filling target list
    # variable names
    DE_entities.extend(["NaN"])
    EC_entities.extend(["NaN"])
    SG_entities.extend(["NaN"])
    RE_extension_name, IE_extension_name, DE_extension_name, SG_extension_name, EC_extension_name, NR_poss_relat = evidence_instance_appending(RE_entities, IE_entities, DE_entities, SG_entities, EC_entities)
    IE_disagg.extend(IE_extension_name)
    RE_disagg.extend(RE_extension_name)
    DE_disagg.extend(DE_extension_name)    
    SG_disagg.extend(SG_extension_name)
    EC_disagg.extend(EC_extension_name)
       

    # indices
    DE_indices.extend(["NaN"])
    EC_indices.extend(["NaN"])
    SG_indices.extend(["NaN"])
    RE_extension_indx, IE_extension_indx, DE_extension_indx, SG_extension_indx, EC_extension_indx, NR_poss_relat  = evidence_instance_appending(RE_indices, IE_indices, DE_indices, SG_indices,EC_indices)
    IE_idx_disagg.extend(IE_extension_indx)
    RE_idx_disagg.extend(RE_extension_indx)
    DE_idx_disagg.extend(DE_extension_indx)
    EC_idx_disagg.extend(EC_extension_indx)
    SG_idx_disagg.extend(SG_extension_indx)
    
    REindicelist = combinations_alllists(RE_indices)
    RE_repetition = int(NR_poss_relat/len(REindicelist))
    Nr_REs_in_instance.extend([len(x) for x in REindicelist] * RE_repetition)
    minimum_RE_idx.extend([min(x) for x in REindicelist] * RE_repetition)
    maximum_RE_idx.extend([max(x) for x in REindicelist] * RE_repetition)
    Multi_RE_Indice_gap.extend([max(x)-min(x) for x in REindicelist] * RE_repetition)

    # combinatory syntactical properties
    # part of same subtree + shortest path between all options
    for i in range(0, len(RE_extension_name)):
        same_tree, missing_var_sametree = check_if_varcombi_same_dependency_subtree(list(chain.from_iterable([ifnotlist_makelist(RE_extension_name[i]), ifnotlist_makelist(IE_extension_name[i]),
                                                  ifnotlist_makelist(DE_extension_name[i]), ifnotlist_makelist(SG_extension_name[i]), 
                                                  #ifnotlist_makelist(ST_extension_name[i]), ifnotlist_makelist(SE_extension_name[i]), 
                                                  ifnotlist_makelist(EC_extension_name[i])
                                                  ])), 
                                                  list(chain.from_iterable([DE_entities, SG_entities, #ST_entities, SE_entities,
                                                  EC_entities])), items_in_subtree_groups)
        same_dependTree.extend(same_tree)
        missing_var_in_same_dependTree.extend([missing_var_sametree])
        minSP, maxSP, sumSP = getShortestPathbetweenPhrases(RE_extension_name[i], IE_extension_name[i], graph)
        RE_IE_minshortpath.extend(minSP)
        RE_IE_maxshortpath.extend(maxSP)
        RE_IE_sumshortpath.extend(sumSP)
        if DE_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(RE_extension_name[i], RE_extension_name[i], graph)
            RE_DE_minshortpath.extend(minSP)
            RE_DE_maxshortpath.extend(maxSP)
            RE_DE_sumshortpath.extend(sumSP)
        else:
            RE_DE_minshortpath.extend(["NaN"])
            RE_DE_maxshortpath.extend(["NaN"])
            RE_DE_sumshortpath.extend(["NaN"])
        if SG_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(RE_extension_name[i], SG_extension_name[i], graph)
            RE_SG_minshortpath.extend(minSP)
            RE_SG_maxshortpath.extend(maxSP)
            RE_SG_sumshortpath.extend(sumSP)
        else:
            RE_SG_minshortpath.extend(["NaN"])
            RE_SG_maxshortpath.extend(["NaN"])
            RE_SG_sumshortpath.extend(["NaN"])
        if EC_extension_name[i]!= "NaN":
            minSP, maxSP, sumSP = getShortestPathbetweenPhrases(RE_extension_name[i], EC_extension_name[i], graph)
            RE_EC_minshortpath.extend(minSP)
            RE_EC_maxshortpath.extend(maxSP)
            RE_EC_sumshortpath.extend(sumSP)
        else:
            RE_EC_minshortpath.extend(["NaN"])
            RE_EC_maxshortpath.extend(["NaN"])
            RE_EC_sumshortpath.extend(["NaN"])
        #if ST_extension_name[i]!= "NaN":
            #minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], ST_extension_name[i], graph)
            #AT_ST_minshortpath.extend(minSP)
            #AT_ST_maxshortpath.extend(maxSP)
            #AT_ST_sumshortpath.extend(sumSP)
        #else:
            #AT_ST_minshortpath.extend(["NaN"])
            #AT_ST_maxshortpath.extend(["NaN"])
            #AT_ST_sumshortpath.extend(["NaN"])
        #if SE_extension_name[i]!= "NaN":
            #minSP, maxSP, sumSP = getShortestPathbetweenPhrases(AT_extension_name[i], SE_extension_name[i], graph)
            #AT_SE_minshortpath.extend(minSP)
            #AT_SE_maxshortpath.extend(maxSP)
            #AT_SE_sumshortpath.extend(sumSP)
        #else:
            #AT_SE_minshortpath.extend(["NaN"])
            #AT_SE_maxshortpath.extend(["NaN"])
            #AT_SE_sumshortpath.extend(["NaN"])

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
    RE_extension_indx_alllists = REindicelist*RE_repetition
    for i in range(0, len(RE_extension_name)):
        indices = list(chain.from_iterable([RE_extension_indx_alllists[i], [IE_extension_indx[i]],[DE_extension_indx[i]], 
                                            ifnotlist_makelist(SG_extension_indx[i]), [EC_extension_indx[i]], 
                                           ]))

        all_before_or_after_split_propos.extend(testIfAllPhrasesBeforeIdx(idx=split_prepos[0], PhraseIndices=indices))
        all_before_or_after_semicolon.extend(testIfAllPhrasesBeforeIdx(idx=semicolon[0], PhraseIndices=indices))

        index_dist_REmin_IE.extend([min(RE_extension_indx_alllists[i]) - IE_extension_indx[i]])
        index_dist_REmax_IE.extend([max(RE_extension_indx_alllists[i]) - IE_extension_indx[i]])

        verb_between_REmin_IE.extend(testVerbInBetween(verb_indx, min(RE_extension_indx_alllists[i]), IE_extension_indx[i]))
        verb_between_REmax_IE.extend(testVerbInBetween(verb_indx, max(RE_extension_indx_alllists[i]), IE_extension_indx[i]))
        if DE_extension_indx[i] != "NaN":
            index_dist_REmin_DE.extend([min(RE_extension_indx_alllists[i]) - DE_extension_indx[i]])
            index_dist_REmax_DE.extend([max(RE_extension_indx_alllists[i]) - DE_extension_indx[i]])

            verb_between_REmin_DE.extend(testVerbInBetween(verb_indx, min(RE_extension_indx_alllists[i]), DE_extension_indx[i]))
            verb_between_REmax_DE.extend(testVerbInBetween(verb_indx, max(RE_extension_indx_alllists[i]), DE_extension_indx[i]))
        else:
            index_dist_REmin_DE.extend(["NaN"])
            index_dist_REmax_DE.extend(["NaN"])
            verb_between_REmin_DE.extend(["NaN"])
            verb_between_REmax_DE.extend(["NaN"])
        if EC_extension_indx[i] != "NaN":
            index_dist_REmin_EC.extend([min(RE_extension_indx_alllists[i]) - EC_extension_indx[i]])
            index_dist_REmax_EC.extend([max(RE_extension_indx_alllists[i]) - EC_extension_indx[i]])

            verb_between_REmin_EC.extend(testVerbInBetween(verb_indx, min(RE_extension_indx_alllists[i]), EC_extension_indx[i]))
            verb_between_REmax_EC.extend(testVerbInBetween(verb_indx, max(RE_extension_indx_alllists[i]), EC_extension_indx[i]))
        else:
            index_dist_REmin_EC.extend(["NaN"])
            index_dist_REmax_EC.extend(["NaN"])
            verb_between_REmin_EC.extend(["NaN"])
            verb_between_REmax_EC.extend(["NaN"])
        if SG_extension_indx[i] != "NaN":
            index_dist_REmin_SGmin.extend([min(RE_extension_indx_alllists[i]) - min(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_REmax_SGmin.extend([max(RE_extension_indx_alllists[i]) - min(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_REmin_SGmax.extend([min(RE_extension_indx_alllists[i]) - max(ifnotlist_makelist(SG_extension_indx[i]))])
            index_dist_REmax_SGmax.extend([max(RE_extension_indx_alllists[i]) - max(ifnotlist_makelist(SG_extension_indx[i]))])

            verb_between_REmin_SGmin.extend(testVerbInBetween(verb_indx, min(RE_extension_indx_alllists[i]), min(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_REmax_SGmin.extend(testVerbInBetween(verb_indx, max(RE_extension_indx_alllists[i]), min(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_REmin_SGmax.extend(testVerbInBetween(verb_indx, min(RE_extension_indx_alllists[i]), max(ifnotlist_makelist(SG_extension_indx[i]))))
            verb_between_REmax_SGmax.extend(testVerbInBetween(verb_indx, max(RE_extension_indx_alllists[i]), max(ifnotlist_makelist(SG_extension_indx[i]))))

            minimum_SG_idx.extend([min(ifnotlist_makelist(SG_extension_indx[i]))])
            maximum_SG_idx.extend([max(ifnotlist_makelist(SG_extension_indx[i]))])
        else:
            index_dist_REmin_SGmin.extend(["NaN"])
            index_dist_REmax_SGmin.extend(["NaN"])
            index_dist_REmin_SGmax.extend(["NaN"])
            index_dist_REmax_SGmax.extend(["NaN"])
            verb_between_REmin_SGmin.extend(["NaN"])
            verb_between_REmax_SGmin.extend(["NaN"])
            verb_between_REmin_SGmax.extend(["NaN"])
            verb_between_REmax_SGmax.extend(["NaN"])
            minimum_SG_idx.extend(["NaN"])
            maximum_SG_idx.extend(["NaN"])

    # Relationship types properties
    # if it contains a negator (not)
    REsent_has_not.extend([1 if any(True if "not" in phrase else False for phrase in RE_entities) else 0] * NR_poss_relat)

    # direct link between association types
    REentitylist = combinations_alllists(RE_entities)
    # Get the length and path
    multiRE_shortestPath_min, multiRE_shortestPath_max = [], []
    has_not = []
    for i in range(0, len(REentitylist)):
        if len(REentitylist[i]) == 1:
            multiRE_shortestPath_min.extend([0])
            multiRE_shortestPath_max.extend([0])
            has_not.extend([1 if "not" in REentitylist[i] else 0])
        else:
            multiRE_shortestPath = []
            words_inlist = list(chain.from_iterable([phrase.split(" ") for phrase in REentitylist[i]]))
            ## checking if any "not" in wordlist
            has_not.extend([1 if "not" in words_inlist else 0])
            for f in range(0, len(words_inlist)):
                for x in range((f + 1), (len(words_inlist))):
                    multiRE_shortestPath.extend([nx.shortest_path_length(graph, source=words_inlist[f].lower(),
                                                                   target=words_inlist[x].lower())])

            multiRE_shortestPath_min.extend([min(multiRE_shortestPath)])
            multiRE_shortestPath_max.extend([max(multiRE_shortestPath)])
    multiRE_shortestPathLen_min.extend(multiRE_shortestPath_min * RE_repetition)
    multiRE_shortestPathLen_max.extend(multiRE_shortestPath_max * RE_repetition)
    REinst_has_not.extend(has_not * RE_repetition)

    # checking ATs part of speech (gramma function)
    fullsentence_RE_POS = complete_evidence_Instances['RE_POS'].iloc[count].split(" ; ")
    wordlengths = [len(word.split(" ")) for word in RE_entities]
    RE_POS_perword = [fullsentence_RE_POS[:wordlengths[0]]]
    for x in range(1,len(wordlengths)):
        idx = (sum(wordlengths[:x]))
        RE_POS_perword.extend([fullsentence_RE_POS[idx:idx+wordlengths[x]]])
    RE_POS_combi = combinations_alllists(RE_POS_perword)
    verb_inside = ["1" if any([True for i in ["VBN", "VB", "VBZ", "VBP", "VBD"] if i in list(chain.from_iterable(RE_POS))]) else "0" for RE_POS in RE_POS_combi]
    noun_inside = ["1" if any([True for i in ["NNS","NN"] if i in list(chain.from_iterable(RE_POS))]) else "0" for RE_POS in RE_POS_combi]
    adj_adv_inside = ["1" if any([True for i in ["JJ","RB", "JJS", "RBS"] if i in list(chain.from_iterable(RE_POS))]) else "0" for RE_POS in RE_POS_combi]
    comp_adj_adv_inside = ["1" if any([True for i in ["JJR","RBR"] if i in list(chain.from_iterable(RE_POS))]) else "0" for RE_POS in RE_POS_combi]
    Verb_in_RE_inst.extend((verb_inside) * RE_repetition)
    Noun_in_RE_inst.extend((noun_inside)* RE_repetition)
    Adj_Adv_in_RE_inst.extend((adj_adv_inside)*RE_repetition)
    Comp_Adj_Adv_in_RE_inst.extend((comp_adj_adv_inside)* RE_repetition)
   

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
    Nr_RE.extend([len(RE_entities)]* NR_poss_relat)
    Nr_DE.extend([len(DE_entities)-1]* NR_poss_relat)
    Nr_IE.extend([len(IE_entities)]* NR_poss_relat)
    Nr_EC.extend([len(EC_entities)-1]* NR_poss_relat)
    Nr_SG.extend([len(SG_entities)-1]* NR_poss_relat)
    joined_DE_instances.extend([DE_joined]* NR_poss_relat)
    Evidence_Truth.extend(["0"]*NR_poss_relat)
    max_sent_REidx.extend([max(RE_indices)] * NR_poss_relat)
    min_sent_REidx.extend([min(RE_indices)] * NR_poss_relat)
    max_sent_IEidx.extend([max(IE_indices)] * NR_poss_relat)
    min_sent_IEidx.extend([min(IE_indices)] * NR_poss_relat)
    if DE_indices != ["NaN"]:
        DE_indices.remove("NaN")
        max_sent_DEidx.extend([max(DE_indices)] * NR_poss_relat)
        min_sent_DEidx.extend([min(DE_indices)] * NR_poss_relat)
    else:
        max_sent_DEidx.extend(["NaN"]*NR_poss_relat)
        min_sent_DEidx.extend(["NaN"]*NR_poss_relat)
    if EC_indices != ["NaN"]:
        EC_indices.remove("NaN")
        max_sent_ECidx.extend([max(EC_indices)] * NR_poss_relat)
        min_sent_ECidx.extend([min(EC_indices)] * NR_poss_relat)
    else:
        max_sent_ECidx.extend(["NaN"]*NR_poss_relat)
        min_sent_ECidx.extend(["NaN"]*NR_poss_relat)
    if SG_indices != ["NaN"]:
        SG_indices.remove("NaN")
        max_sent_SGidx.extend([max(SG_indices)] * NR_poss_relat)
        min_sent_SGidx.extend([min(SG_indices)] * NR_poss_relat)
    else:
        max_sent_SGidx.extend(["NaN"]*NR_poss_relat)
        min_sent_SGidx.extend(["NaN"]*NR_poss_relat)
   
    # check if all lists same length
    print_summary_of_length_ofLists([IE_disagg, RE_disagg, DE_disagg, SG_disagg, EC_disagg, SENT_disagg, DOI_disagg, sent_ID_disagg,
                                     IE_idx_disagg, RE_idx_disagg, DE_idx_disagg, EC_idx_disagg, SG_idx_disagg, 
                                     split_prepos_idx_disagg,Nr_RE, Nr_DE, Nr_IE, Nr_EC, Nr_SG,Nr_verbs, earliestverb_indx,
                                     latestverb_indx, Verb_in_RE_inst, Noun_in_RE_inst, Adj_Adv_in_RE_inst, Comp_Adj_Adv_in_RE_inst,
                                     Nr_REs_in_instance, Multi_RE_Indice_gap, minimum_RE_idx, maximum_RE_idx, multiRE_shortestPathLen_min,
                                     multiRE_shortestPathLen_max, same_dependTree, missing_var_in_same_dependTree, joined_DE_instances,
                                     REinst_has_not, REsent_has_not, RE_IE_minshortpath, RE_IE_maxshortpath,
                                     RE_IE_sumshortpath, RE_DE_minshortpath, RE_DE_maxshortpath, RE_DE_sumshortpath, RE_SG_minshortpath,
                                     RE_SG_maxshortpath, RE_SG_sumshortpath, RE_EC_minshortpath, RE_EC_maxshortpath, RE_EC_sumshortpath,
                                     #AT_ST_minshortpath, AT_ST_maxshortpath, AT_ST_sumshortpath,
                                     #AT_SE_minshortpath, AT_SE_maxshortpath, AT_SE_sumshortpath,
                                     all_before_or_after_split_propos, index_dist_REmin_IE, index_dist_REmax_IE, index_dist_REmin_DE,
                                     index_dist_REmax_DE, index_dist_REmin_EC, index_dist_REmax_EC, index_dist_REmin_SGmin, index_dist_REmin_SGmax,
                                     index_dist_REmax_SGmin, index_dist_REmax_SGmax,
                                     #index_dist_REmin_STmin, index_dist_REmin_STmax, index_dist_ATmax_STmin, index_dist_ATmax_STmax,
                                     #index_dist_ATmin_SEmin, index_dist_ATmin_SEmax, index_dist_ATmax_SEmin, index_dist_ATmax_SEmax,
                                     verb_between_REmin_IE, verb_between_REmax_IE, verb_between_REmin_DE, verb_between_REmax_DE, verb_between_REmin_EC,
                                     verb_between_REmax_EC, verb_between_REmin_SGmin, verb_between_REmax_SGmin, verb_between_REmin_SGmax, verb_between_REmax_SGmax,
                                     #verb_between_ATmin_STmin, verb_between_ATmax_STmin, verb_between_ATmin_STmax, verb_between_ATmax_STmax,
                                     #verb_between_ATmin_SEmin, verb_between_ATmax_SEmin, verb_between_ATmin_SEmax, verb_between_ATmax_SEmax,
                                     max_sent_REidx, min_sent_REidx, max_sent_SGidx, min_sent_SGidx,
                                     max_sent_IEidx, min_sent_IEidx, max_sent_DEidx, min_sent_DEidx, max_sent_ECidx, min_sent_ECidx,
                                     max_sent_SGidx, min_sent_SGidx, 
                                     #max_sent_STidx, min_sent_STidx, max_sent_SEidx, min_sent_SEidx,
                                     semicolon_idx, all_before_or_after_semicolon, max_negator_idx, min_negator_idx])



possible_evidence_instances = pd.DataFrame({'DOI': DOI_disagg, 'Sentence': sent_ID_disagg, 'Evidence_Truth': Evidence_Truth, 'Fullsentence': SENT_disagg,
                                                 'DirectEffects': DE_disagg, 'IndirectEffects': IE_disagg,
                                                 'Relationship': RE_disagg, 'StudyGroup': SG_disagg, #'StudyType': ST_disagg, 'StudyEnvironment': SE_disagg, 
                                                 'EcoConsequences': EC_disagg,
                                                 'IE_idx': IE_idx_disagg, 'DE_idx': DE_idx_disagg,
                                                 'EC_idx': EC_idx_disagg, 'Split_Propos_idx': split_prepos_idx_disagg,
                                                 'all_before_or_after_split_propos': all_before_or_after_split_propos,
                                                 'semicolon_idx': semicolon_idx, 'all_before_or_after_semicolon': all_before_or_after_semicolon,
                                                 'max_negator_idx': max_negator_idx, 'min_negator_idx': min_negator_idx,
                                                 'index_dist_REmin_IE': index_dist_REmin_IE, 'index_dist_REmax_IE': index_dist_REmax_IE,
                                                 'index_dist_REmin_DE': index_dist_REmin_DE, 'index_dist_REmax_DE': index_dist_REmax_DE,
                                                 'index_dist_REmin_EC': index_dist_REmin_EC, 'index_dist_REmax_EC': index_dist_REmax_EC,
                                                 'index_dist_REmin_SGmin': index_dist_REmin_SGmin,  'index_dist_REmax_SGmin': index_dist_REmax_SGmin,
                                                 'index_dist_REmin_SGmax': index_dist_REmin_SGmax, 'index_dist_REmax_SGmax': index_dist_REmax_SGmax,
                                                 'same_dependTree': same_dependTree, 'missing_var_in_same_dependTree' : missing_var_in_same_dependTree,  'joined_DE_instances': joined_DE_instances,
                                                 'Nr_RE': Nr_RE, 'Nr_DE': Nr_DE, 'Nr_IE': Nr_IE, 'Nr_EC': Nr_EC, 'Nr_SG': Nr_SG, #'Nr_ST': Nr_ST, 'Nr_SE': Nr_SE,
                                                 'Verb_in_RE_inst': Verb_in_RE_inst, 'Noun_in_RE_inst': Noun_in_RE_inst,
                                                 'Adj_Adv_in_RE_inst': Adj_Adv_in_RE_inst, 'Comp_Adj_Adv_in_RE_inst': Comp_Adj_Adv_in_RE_inst,
                                                 'Nr_REs_in_instance': Nr_REs_in_instance, 'Multi_RE_Indice_gap': Multi_RE_Indice_gap,
                                                 'minimum_RE_idx': minimum_RE_idx, 'maximum_RE_idx': maximum_RE_idx,
                                                 'minimum_SG_idx': minimum_SG_idx, 'maximum_SG_idx': maximum_SG_idx,
                                                 'REinst_has_not': REinst_has_not, 'REsent_has_not': REsent_has_not,
                                                 'Nr_verbs': Nr_verbs, 'earliestverb_indx': earliestverb_indx, 'latestverb_indx': latestverb_indx,
                                                 'multiRE_shortestPathLen_min': multiRE_shortestPathLen_min,'multiRE_shortestPathLen_max': multiRE_shortestPathLen_max,
                                                 'RE_IE_minshortpath': RE_IE_minshortpath, 'RE_IE_maxshortpath': RE_IE_maxshortpath, 'RE_IE_sumshortpath': RE_IE_sumshortpath,
                                                 'RE_DE_minshortpath': RE_DE_minshortpath, 'RE_DE_maxshortpath': RE_DE_maxshortpath, 'RE_DE_sumshortpath': RE_DE_sumshortpath,
                                                 'RE_SG_minshortpath': RE_SG_minshortpath, 'RE_SG_maxshortpath': RE_SG_maxshortpath, 'RE_SG_sumshortpath': RE_SG_sumshortpath,
                                                 'RE_EC_minshortpath': RE_EC_minshortpath, 'RE_EC_maxshortpath': RE_EC_maxshortpath, 'RE_EC_sumshortpath': RE_EC_sumshortpath,
                                                 'verb_between_REmin_IE': verb_between_REmin_IE, 'verb_between_REmax_IE': verb_between_REmax_IE,
                                                 'verb_between_REmin_DE': verb_between_REmin_DE, 'verb_between_REmax_DE': verb_between_REmax_DE,
                                                 'verb_between_REmin_EC': verb_between_REmin_EC, 'verb_between_REmax_EC': verb_between_REmax_EC,
                                                 'verb_between_REmin_SGmin': verb_between_REmin_SGmin, 'verb_between_REmax_SGmin': verb_between_REmax_SGmin,
                                                 'verb_between_REmin_SGmax': verb_between_REmin_SGmax, 'verb_between_REmax_SGmax': verb_between_REmax_SGmax,
                                                 'max_sent_REidx': max_sent_REidx, 'min_sent_REidx': min_sent_REidx,
                                                 'max_sent_IEidx': max_sent_IEidx, 'min_sent_IEidx': min_sent_IEidx,
                                                 'max_sent_DEidx': max_sent_DEidx, 'min_sent_DEidx': min_sent_DEidx,
                                                 'max_sent_ECidx': max_sent_ECidx, 'min_sent_ECidx': min_sent_ECidx,
                                                 'max_sent_SGidx': max_sent_SGidx, 'min_sent_SGidx': min_sent_SGidx,
                                                  })

possible_evidence_instances.replace({'NaN': -100}, regex=False, inplace=True)

possible_evidence_instances.to_csv("possible_evidence_instances4.csv", index=False)
#possible_evidence_instances.to_csv("possible_evidence_instances_Manual4.csv", index=False)
# #
print("Nr of unique articles contributing evidence: ",len(set(possible_evidence_instances['DOI'])))