import os
import pandas as pd
import numpy as np
import nltk

###########################################################################
## This script transforms the labeled articles into a relational dataframe
## of list of phrases per label per sentence per article
###########################################################################

# Functions
def IOB_sequence_tracing(B_indx_list, I_index_list, Words_data):
    """Identify phrases based on continues labeling of word sequences with the same tag
       for IOB labeling."""
    tags = []
    for indx in B_indx_list:
        tag_string = str(Words_data.iloc[indx])
        if indx + 1 in I_index_list:
            tag_string = tag_string + " " + str(Words_data.iloc[indx + 1])
            nr_concatenated_words = 1
            for x in range(2, len(I_index_list)+1):
                if all(elem in I_index_list  for elem in range(indx + 1, indx + x+1)):
                    tag_string = tag_string + " " + str(Words_data.iloc[indx + x])
                    nr_concatenated_words = x
            I_index_list = [I_index_list.remove(indx + w) for w in range(1,nr_concatenated_words)]
        tags.append(tag_string)
    I_index_list = [i for i in I_index_list if i]
    if bool(I_index_list):
        for I_indx in I_index_list:
            tags.append(Words_data.iloc[I_indx])
    [tags.remove(el) for el in ["-", "the", "a", "s"] if el in tags]
    return tags

def IO_sequence_tracing(indx_list, Words_data):
    """Identify phrases based on continues labeling of word sequences with the same tag
       for IO labeling."""
    tags = []
    if len(indx_list) > 1:
        already_included = -1
        for indx in indx_list:
            if indx > already_included:
                tag_string = str(Words_data.iloc[indx])
                if indx + 1 in indx_list:
                    tag_string = tag_string + " " + str(Words_data.iloc[indx + 1])
                    already_included = indx + 1
                    for x in range(2, len(indx_list)+1):
                        if all(elem in indx_list for elem in range(indx + 1, indx + x+1)):
                            tag_string = tag_string + " " + str(Words_data.iloc[indx + x])
                            already_included = indx + x
                tags.append(tag_string)
    else:
        if bool(indx_list):
            tags.append(Words_data.iloc[indx_list[0]])
    [tags.remove(el) for el in ["-", "the", "a", "s"] if el in tags]
    return tags


def IO_sequence_tracingopt(indx_list, Words_data):
    """Identify phrases based on continues labeling of word sequences with the same tag
       for IO labeling. This function is designed to be used dynamically in other functions."""
    tags = []
    if len(indx_list) > 1:
        nested_worldlength = [indx_list]
        n_long_words = [i for i in indx_list if i + 1 in indx_list]
        nested_worldlength.append(n_long_words)
        for n in range(2,15):
            n_long_words = [i for i in n_long_words if i+n in indx_list]
            if bool(n_long_words):
                nested_worldlength.append(n_long_words)
        for n in range(len(nested_worldlength)-1, 0, -1):
            for indx in nested_worldlength[n]:
                tags.append(" ".join(Words_data[indx:(indx+n+1)]))
            for x in range(0,n):
                nested_worldlength[x] = [e for e in nested_worldlength[x] if e not in nested_worldlength[n]]
                for g in range(1, int(n-x+1)):
                    nested_worldlength[x] = [e for e in nested_worldlength[x] if e not in list(np.asarray(nested_worldlength[n]) + g)]
        if len(nested_worldlength[0])>0:
            tags.extend(list(Words_data[nested_worldlength[0]]))
    else:
        if bool(indx_list):
            tags.append(Words_data.iloc[indx_list[0]])
    # print(tags)
    [tags.remove(el) for el in ["-", "the", "a", "s"] if el in tags]
    return tags


def extendTagsToAllEqualWordSeq(dataframe):
    """Extend labels of all phrases to equivalent phrases in the rest of the document."""
    labels = list(dict.fromkeys(dataframe['Tag']))
    labels.remove('O')
    for label in labels:
        label_indices = list(np.where(dataframe['Tag'] == label)[0])
        unique_labeled_words = list(dict.fromkeys(IO_sequence_tracingopt(indx_list=label_indices, Words_data=dataframe['Word'])))
        unique_labeled_words = unique_labeled_words.replace("[SEP]", "").replace("[CLS]", "")
        for word in unique_labeled_words:
            words = word.split()
            if len(words) == 1:
                dataframe['Tag'].iloc[list(np.where((dataframe['Word'] == word) & (dataframe['Tag'] == 'O'))[0])] = label
            else:
                first_word_index = list(np.where((dataframe['Word'] == words[0]) & (dataframe['Tag'] == 'O'))[0])
                for indx in first_word_index:
                    if list(dataframe['Word'].iloc[indx:(indx+len(words))]) == words:
                        dataframe['Tag'].iloc[indx:(indx+len(words))] = label
    return dataframe

def extendSpecificTagsToAllEqualWordSeq(dataframe, Tagname):
    """Extend all phrases of a specific Label to equivalent phrases in the rest of the document."""
    label_indices = list(np.where(dataframe['Tag'] == Tagname)[0])
    unique_labeled_words = list(dict.fromkeys(IO_sequence_tracingopt(indx_list=label_indices, Words_data=dataframe['Word'])))
    [unique_labeled_words.remove(el) for el in ["-", "the", "a"] if el in unique_labeled_words]
    for word in unique_labeled_words:
        words = word.split()
        if len(words) == 1:
            dataframe.loc[list(np.where((dataframe['Word'] == word) & (dataframe['Tag'] == 'O'))[0]), 'Tag'] = Tagname
        else:
            first_word_index = list(np.where((dataframe['Word'] == words[0]) & (dataframe['Tag'] == 'O'))[0])
            for indx in first_word_index:
                if list(dataframe.loc[indx:(indx+len(words)), 'Word']) == words:
                    dataframe.loc[indx:(indx+len(words)), 'Tag'] = Tagname
    return dataframe


def addPOS(dataframe):
    """ Add a part of speech tag to a list of words in sentences. """
    POS = list(nltk.pos_tag([str(word) for word in dataframe['Word']]))
    POS = pd.DataFrame(data=POS, columns=["Word", "POS_tag"])
    dataframe['POS'] = POS["POS_tag"]
    dataframe.loc[list(np.where((dataframe['Word'] == "[SEP]") | (dataframe['Word'] == "[CLS]"))[0]), 'POS'] = 'PAD'
    dataframe.loc[list(np.where((dataframe['Word'] == "[") | (dataframe['Word'] == "]")| (dataframe['Word'] == "(") | (dataframe['Word'] == ")"))[0]), 'POS'] = '.'
    dataframe.loc[list(np.where((dataframe['Word'] == "whereas") | (dataframe['Word'] == "while")| (dataframe['Word'] == "unlike") | (dataframe['Word'] == "although")|(dataframe['Word'] == "though")| (dataframe['Word'] == "such"))[0]), 'POS'] = 'IN'
    return dataframe

def extendVariableNamesToNeighboringAdjectNouns(dataframe, Tagnames):
    """Extend Labeled Phrases to neighboring adjectives or nouns to be sure to capture the whole variable name."""
    for Tagname in Tagnames:
        for turn in [1,2,3,4]:
            label_indices = list(np.where(dataframe['Tag'] == Tagname)[0])
            label_indices_post = [i+1 for i in label_indices]
            label_indices_pre = [i-1 for i in label_indices]
            x = list(np.where(dataframe.loc[label_indices_post, 'Tag'] == 'O')[0])
            f = list(np.where(dataframe.loc[label_indices_pre, 'Tag'] == 'O')[0])
            if Tagname == "I-DirectEffects":
                y = [i for i in label_indices_post if (dataframe['POS'].iloc[i] in ["NN", "NNS", "JJ", "JJS"]) or (dataframe['Word'].iloc[i] in ["of", "over", "/", "the"])]
                g = [i for i in label_indices_pre if ((dataframe['POS'].iloc[i] in ["NN", "NNS", "JJ", "JJS", "VBN", "JJR", "VBD"]) or (dataframe['Word'].iloc[i] in ["of", "over", "/", "the", "-"])) and (dataframe['Word'].iloc[i] != "s")]
            else:
                y = [label_indices_post[i] for i in x if (dataframe['POS'].iloc[label_indices_post[i]] in ["NN", "NNS", "JJ", "JJS"])]
                g = [label_indices_pre[i] for i in f if (dataframe['POS'].iloc[label_indices_pre[i]] in ["NN", "NNS", "JJ", "JJS"]) and (dataframe['Word'].iloc[label_indices_pre[i]] != "s")]
            dataframe.loc[y, 'Tag'] = Tagname
            dataframe.loc[g, 'Tag'] = Tagname
            label_indices = list(np.where(dataframe['Tag'] == Tagname)[0])
            open_hyphens = [i-1 for i in label_indices if (dataframe.loc[i, 'Word'] == '-') and (i-1 not in label_indices)]
            dataframe.loc[open_hyphens, 'Tag'] = Tagname
            open_hyphens = [i+1 for i in label_indices if (dataframe.loc[i, 'Word'] == '-') and (i+1 not in label_indices)]
            dataframe.loc[open_hyphens, 'Tag'] = Tagname
            open_slash = [i-1 for i in label_indices if (dataframe.loc[i, 'Word'] == '/') and (i-1 not in label_indices)]
            dataframe.loc[open_slash ,'Tag'] = Tagname
            open_slash = [i+1 for i in label_indices if (dataframe.loc[i, 'Word'] == '/') and (i+1 not in label_indices)]
            dataframe.loc[open_slash, 'Tag'] = Tagname
            y = [i for i in label_indices_post if ((dataframe.loc[i-1, 'POS'] in ["JJ", "JJS"]) or (
                        "," in dataframe.loc[i-4:i, 'Word'])) and (dataframe.loc[i, 'Word'] in ["and", "or"]) and (dataframe.loc[i+1, 'Tag'] == Tagname)]
            dataframe.loc[y, 'Tag'] = Tagname
            y = [i for i in label_indices_post if (dataframe.loc[i - 1, 'POS'] in ["NN", "NNS"]) and (dataframe.loc[i, 'Word'] in ["and", "or"]) and (
                             dataframe.loc[i + 1, 'Tag'] == Tagname) and (((
                             dataframe.loc[i + 2, 'Tag'] == Tagname) and (dataframe.loc[i-2, 'Tag'] != Tagname)) or ((
                             dataframe.loc[i - 2, 'Tag'] == Tagname) and (dataframe.loc[i+2, 'Tag'] != Tagname)))]
            dataframe.loc[y, 'Tag'] = Tagname
            y = [i for i in label_indices_post if (
                        "," in dataframe.loc[i-3:i, 'Word']) and ((
                        "," in dataframe.loc[i+1:i+4, 'Word']) or (
                        "." in dataframe.loc[i+1:i+4, 'Word'])) and (dataframe.loc[i, 'Word'] in ["and", "or"]) and (
                             dataframe.loc[i + 1, 'Tag'] == Tagname) and ((
                             dataframe.loc[i + 2, 'Tag'] == Tagname) or (
                             dataframe.loc[i - 2, 'Tag'] == Tagname))]
            dataframe.loc[y, 'Tag'] = Tagname
    return dataframe

def extendAssociationTypesToNeighboringAdjectNegatives(dataframe):
    """Extend Association Type phrases to neighboring adjectives and negatives."""
    for turn in [1,2,3]:
        label_indices = list(np.where(dataframe['Tag'] == 'I-Relationship')[0])
        label_indices_post = [i+1 for i in label_indices]
        label_indices_pre = [i-1 for i in label_indices]
        x = list(np.where(dataframe.loc[label_indices_post, 'Tag'] == 'O')[0])
        f = list(np.where(dataframe.loc[label_indices_pre,'Tag'] == 'O')[0])
        g = [label_indices_pre[i] for i in f if (dataframe.loc[label_indices_pre[i], 'Word'] in ["non", "not", "Non", "Not", "no", "No", "nil", "relationship", "relationships", "effect", "impact"]) or (dataframe['POS'].iloc[label_indices_pre[i]] in ["JJ", "JJS", "JJR"])]
        y = [label_indices_post[i] for i in x if (dataframe.loc[label_indices_post[i], 'Word'] in ["no", "not", "No", "Not", "nil", "relationship", "relationships", "effect", "impact"]) or (dataframe['POS'].iloc[label_indices_post[i]] in ["JJ", "JJS", "JJR"])]
        dataframe.loc[y, 'Tag'] = 'I-Relationship'
        dataframe.loc[g, 'Tag'] = 'I-Relationship'
        label_indices_post = [i + 2 for i in label_indices]
        label_indices_pre = [i - 2 for i in label_indices]
        x = list(np.where(dataframe.loc[label_indices_post, 'Tag'] == 'O')[0])
        f = list(np.where(dataframe.loc[label_indices_pre, 'Tag'] == 'O')[0])
        g = [label_indices_pre[i] for i in f if (dataframe.loc[label_indices_pre[i], 'Word'] in ["non", "not", "Non", "Not", "no", "No", "Lack", "lack", "nil", "relationship", "relationships", "effect", "impact"]) or (
                dataframe.loc[label_indices_pre[i], 'POS'] in ["JJ", "JJS", "JJR"])]
        y = [label_indices_post[i] for i in x if (dataframe.loc[label_indices_post[i], 'Word'] in ["no", "not", "No", "Not", "nil", "relationship", "relationships", "effect", "impact"])]
        dataframe.loc[y, 'Tag'] = 'I-Relationship'
        dataframe.loc[g, 'Tag'] = 'I-Relationship'
        label_indices = list(np.where(dataframe['Tag'] == 'I-Relationship')[0])
        assoc_sent = [count for count, value in enumerate(dataframe['Sentence']) if (value in list(set(dataframe.loc[label_indices, 'Sentence']))) and dataframe.loc[count, 'Tag'] == "O"]
        y = [i for i in assoc_sent if str(dataframe.loc[i, 'Word']).lower() in ["evidence", "not", "no", "nil", "relationship", "relationships", "effect", "influence", "impact", "positive", "negative", "mixed", "high"]]
        dataframe.loc[y, 'Tag'] = 'I-Relationship'
    return dataframe

def LabelAssocTypeWords(labeled_data):
    """Label words that are obviously describing statistical associations based on keywords."""
    x = list(np.where((labeled_data['Word'] == 'significant') | (
                            labeled_data['Word'] == 'insignificant') | (
                            labeled_data['Word'] == 'consistent') | (
                            labeled_data['Word'] == 'Consistent') | (
                            labeled_data['Word'] == 'associated') | (
                            labeled_data['Word'] == 'inconsistent') | (
                            labeled_data['Word'] == 'Associations') | (
                            labeled_data['Word'] == 'associations') | (
                            labeled_data['Word'] == 'Association') | (
                            labeled_data['Word'] == 'association') | (
                            labeled_data['Word'] == 'significantly') | (
                            labeled_data['Word'] == 'insignificantly') | (
                            labeled_data['Word'] == 'influence'))[0])
    labeled_data.loc[x, 'Tag'] = 'I-Relationship'
    return labeled_data

def ExtendATwordsToNegation(labeled_data):
    """Identify linked negation words and label them to make sure the correct association type is identified."""
    AT_words = list(np.where(labeled_data['Tag'] == 'I-Relationship')[0])
    x = []
    x.extend(idx - 1 for idx in AT_words if
             labeled_data.loc[idx - 1, 'Word'] in ["non", "not", "Non", "Not", "no", "No", "nil"])
    x.extend(idx - 2 for idx in AT_words if
             labeled_data.loc[idx - 2, 'Word'] in ["non", "not", "Non", "Not", "no", "No", "Lack", "lack", "nil"])
    labeled_data.loc[x, 'Tag'] = 'I-Relationship'
    return labeled_data

def FillLinkingATs(labeled_data):
    """Identify linking words between 2 AT phrases and label them to unify the phrase."""
    AT_words = list(np.where(labeled_data['Tag'] == 'I-Relationship')[0])
    x = [x+1 for x in AT_words if (x+2 in AT_words) and (labeled_data.loc[x+1, 'Word'] in ["of", ",", "and", "for", "to", "a", "the"])]
    x.extend((x+2 for x in AT_words if (x+3 in AT_words) and (labeled_data.loc[x+1, 'Word'] in ["of", ",", "and", "for", "to", "a", "the"]) and (labeled_data['Word'].iloc[x+2] in ["of", ",", "and", "for", "to", "a", "the"])))
    labeled_data.loc[x, 'Tag'] = 'I-Relationship'
    return labeled_data


def UniqueLabelsFreq(phraselist):
    """Find unique Labels and count their frequency"""
    Labels, freq_Labels = [], []
    for instance in phraselist:
        Labels.extend(instance.split(" ; "))
    unique_Labels = list(dict.fromkeys(Labels))
    for Label in unique_Labels:
        freq_Labels.append(Labels.count(Label))
    return unique_Labels, freq_Labels


def FindIndexesPerLabel_IO(labeled_data, sentence_idx_range):
    """ Composes lists of indices per label for IO labeling. """
    RE_indx = list(np.where(labeled_data.loc[sentence_idx_range, 'Tag'] == "I-Relationship")[0])
    DE_indx = list(np.where(labeled_data.loc[sentence_idx_range, 'Tag'] == "I-DirectEffects")[0])
    EC_indx = list(np.where(labeled_data.loc[sentence_idx_range, 'Tag'] == "I-EcoConsequences")[0])
    IE_indx = list(np.where(labeled_data.loc[sentence_idx_range, 'Tag'] == "I-IndirectEffects")[0])
    SG_indx = list(np.where(labeled_data.loc[sentence_idx_range, 'Tag'] == "I-StudyGroup")[0])
    return RE_indx, DE_indx, EC_indx, IE_indx, SG_indx



# Execution

############ Abbreviation list
## ST = study, DE = DirectEffects, EC = Ecological Consequences, RE = Relationship, SG = Study Group, IE = Indirect Effects


# Settings
# mode = "IOB"
mode = "IO"
TagToWordExtension = False     # Do you want that labeled phrases are extended to all equivalent words in the rest of the document?
manual_label = True     # Do you want to summarize the manually labeled data

datafolder = r"C:\Users\4209416\OneDrive - Universiteit Utrecht\Desktop\ETAIN\PHIA_framework\Automated-KG\newCrossRef\Indirect_Effects\final_kg"
os.chdir(datafolder)
if manual_label == True:
    listOfFiles = os.listdir(path=os.path.join(os.getcwd(), "Manually_labelled"))
else:
    listOfFiles = os.listdir(path=os.path.join(os.getcwd(), "Predict_labelled_clean"))
    print(listOfFiles)

instance_RE, instance_DE, instance_EC, instance_IE, instance_SG, instance_ST, sentenceid, fullsentence, RE_POS, DE_POS, IE_POS, sentence_POS = [], [], [], [], [], [], [], [], [], [], [], []
len_instance_before = 0
for file in listOfFiles:
    print("Processing file: ", file)
    if manual_label == True:
        labeled_data = pd.read_csv(os.path.join(os.getcwd(), ("Manually_labelled/" + file)), encoding="latin1")
    else:
        labeled_data = pd.read_csv(os.path.join(os.getcwd(), ("Predict_labelled_clean/" + file)), encoding="latin1")

    # Sometimes [SEP] and [CLS] are wrongly assigned a label.
    labeled_data.loc[list(np.where((labeled_data['Word'] == "[SEP]") | (labeled_data['Word'] == "[CLS]") | (labeled_data['Word'] == ".") | (labeled_data['Word'] == "s")| (labeled_data['Word'] == "'"))[0]),'Tag'] = 'O'

    # we extend our data with part of speech tags
    labeled_data = addPOS(labeled_data)

    # we extend the association types to relevant neighbors and keywords
    labeled_data = LabelAssocTypeWords(labeled_data)
    labeled_data = ExtendATwordsToNegation(labeled_data)
    labeled_data = FillLinkingATs(labeled_data)

    # we extent the other labels to full variable name phrases
    labeled_data = extendVariableNamesToNeighboringAdjectNouns(labeled_data, ["I-DirectEffects", "I-EcoConsequences","I-IndirectEffects"])
    labeled_data = extendAssociationTypesToNeighboringAdjectNegatives(labeled_data)
    if TagToWordExtension:
        labeled_data = extendTagsToAllEqualWordSeq(labeled_data)
    # labeled_data = extendSpecificTagsToAllEqualWordSeq(labeled_data, "I-HealthEffects")
    labeled_data = extendSpecificTagsToAllEqualWordSeq(labeled_data, "I-DirectEffects")

    ## Composing relational dataset of phrases per label per sentence
    sentences = list(dict.fromkeys(labeled_data['Sentence']))
    for sentence in sentences:
        sentence_idx_range = np.where(labeled_data['Sentence'] == sentence)[0]
        sent_POS = labeled_data['POS'].iloc[sentence_idx_range]
        if any(labeled_data['Tag'].iloc[sentence_idx_range] != 'O'):
            if mode == "IO":
                # if IOB has not sufficient quality to distinguish ingroup classification, or IO labeling was used
                RE_indx, DE_indx, EC_indx, IE_indx, SG_indx= FindIndexesPerLabel_IO(labeled_data, sentence_idx_range)
                instance_RE.append(" ; ".join(IO_sequence_tracing(indx_list=RE_indx, Words_data=labeled_data['Word'].iloc[sentence_idx_range])))
                instance_DE.append(" ; ".join(IO_sequence_tracing(indx_list=DE_indx, Words_data=labeled_data['Word'].iloc[sentence_idx_range])))
                instance_EC.append(" ; ".join(IO_sequence_tracing(indx_list=EC_indx, Words_data=labeled_data['Word'].iloc[sentence_idx_range])))
                instance_IE.append(" ; ".join(IO_sequence_tracing(indx_list=IE_indx, Words_data=labeled_data['Word'].iloc[sentence_idx_range])))
                instance_SG.append(" ; ".join(IO_sequence_tracing(indx_list=SG_indx, Words_data=labeled_data['Word'].iloc[sentence_idx_range])))
                RE_POS.append(" ; ".join(sent_POS.iloc[RE_indx]))
                DE_POS.append(" ; ".join(sent_POS.iloc[DE_indx]))
                IE_POS.append(" ; ".join(sent_POS.iloc[IE_indx]))
            sentenceid.append(sentence)
            sentence_POS.append(" ; ".join(sent_POS[1:-1]))
            sentence_txt = " ".join(str(e) for e in labeled_data.loc[sentence_idx_range, 'Word'])
            fullsentence.append(sentence_txt.replace("[CLS]", "").replace("[SEP]", ""))

    instance_ST.extend([file] * (len(instance_RE)-len_instance_before))
    len_instance_before = len(instance_RE)

## ST = study, IE = IndirectEffects, DE = DirectEffects, RE = Relationship, SG = Study Group, EC = Ecological Consequences

print("Number of instances: ", len(instance_RE))
print("Number of sentences: ", len(sentenceid))
print("Number of full sentences: ", len(fullsentence))
print("Number of RE: ", len(instance_RE))
print("Number of DE: ", len(instance_DE))
print("Number of EC: ", len(instance_EC))
print("Number of IE: ", len(instance_IE))
print("Number of SG: ", len(instance_SG))
print("Number of sentence_POS: ", len(sentence_POS))
print("Number of RE_POS: ", len(RE_POS))
print("Number of DE_POS: ", len(DE_POS))
print("Number of IE_POS: ", len(IE_POS))


# make dataframe
Evidence_instances_df = pd.DataFrame({'DOI': instance_ST, 'Sentence': sentenceid, 
                                      'DirectEffects': instance_DE, 'Relationship': instance_RE, 
                                      'StudyGroup': instance_SG, 
                                      'EcoConsequences': instance_EC,'IndirectEffects': instance_IE, 
                                      'Fullsentence': fullsentence, 'IE_POS': IE_POS, 'DE_POS': DE_POS, 
                                      'RE_POS': RE_POS, 'sentence_POS': sentence_POS })
print(Evidence_instances_df.head())

# write dataframe
if TagToWordExtension:
    file_name_suffix = "_enhanced"
else:
    file_name_suffix = ""

if manual_label:
    file_name_suffix2 = "_ManualLabel"
else:
    file_name_suffix2 = ""

csv = os.path.join(os.getcwd(), ("Evidence_instances_df"+ file_name_suffix2 +file_name_suffix + ".csv"))
Evidence_instances_df.to_csv(csv, index=False)

complete_evidence_Instances = Evidence_instances_df.iloc[list(np.where((Evidence_instances_df['IndirectEffects'] != "") & (Evidence_instances_df['Relationship'] != ""))[0])]
print(complete_evidence_Instances.head())
csv = os.path.join(os.getcwd(), ("Complete_evidence_Instances"+ file_name_suffix2 +file_name_suffix + ".csv"))
complete_evidence_Instances.to_csv(csv, index=False)

# Write the unique labels and their frequency for modified columns
if manual_label == False:
    # For 'EcoConsequences'
    unique_Eco, freq_Eco = UniqueLabelsFreq(complete_evidence_Instances['EcoConsequences'])
    pd.DataFrame({'EcoConsequences': unique_Eco, 'Freq': freq_Eco}).to_csv(
        os.path.join(os.getcwd(), ("unique_EcoConsequences.csv")), index=False)

    # For 'IndirectEffects'
    unique_IE, freq_IE = UniqueLabelsFreq(complete_evidence_Instances['IndirectEffects'])
    pd.DataFrame({'IndirectEffects': unique_IE, 'Freq': freq_IE}).to_csv(
        os.path.join(os.getcwd(), ("unique_IndirectEffects.csv")), index=False)

    # For 'Relationship'
    unique_Rel, freq_Rel = UniqueLabelsFreq(complete_evidence_Instances['Relationship'])
    pd.DataFrame({'Relationship': unique_Rel, 'Freq': freq_Rel}).to_csv(
        os.path.join(os.getcwd(), ("unique_Relationship.csv")), index=False)


