# author: Ziyue (Julia) Wang
# last updated: 2022 December 3


######## import required modules ########

import pandas as pd
import re
import numpy as np
import os

####### get files #######

# read_conll takes in a subcorpus conll file and outputs a dataframe
def read_conll(filepath):
    # read in the conll file
    data = pd.read_fwf(filepath)
    
   # deal with files with more than one line of comment in the beginning
    k = 0
    if '#' in data.columns:
        while data.iloc[k, 1][0] == '#':
            data = data.drop(k).reset_index(drop=True)
    else:
        while data.iloc[k, 0][0] == '#':
            data = data.drop(k).reset_index(drop=True)
    
    col_names = ['WORD', 'SEGM', 'POS', 'MORPH', 'HEAD', 'EDGE', 'MISC']
    col_names_full = ['ID', 'WORD', 'SEGM', 'POS', 'MORPH', 'HEAD', 'EDGE', 'MISC']

    # update the clean dataframe
    if '#' in data.columns: # deal with files that automatically get its ID read into a separate column
        # extract useful information from the conll file
        useful_info = data.iloc[:, 1].str.split('\t')
        # put the useful information (except for ID) into a dataframe
        cleaned_data = pd.DataFrame(columns = col_names, data = useful_info.tolist()).fillna('_')
        # add ID into that dataframe
        cleaned_data['ID'] = data['#']
        # reorder the columns of the dataframe
        cleaned_data = cleaned_data[col_names_full]
    
    else:
        # extract useful information from the conll file
        useful_info = data.iloc[:, 0].str.split('\t')
        # put the useful information into a dataframe
        cleaned_data = pd.DataFrame(columns = col_names_full, data = useful_info.tolist()).fillna('_')

    return cleaned_data


# extract_id takes an ATF converted file and extracts the ID column as a vector
def extract_id(filepath):
    # read in the conll file converted by the ATF converter
    data = pd.read_fwf(filepath)
    useful_info = data.iloc[1:, 0].str.split('\t')
    # get only the id
    id_col = [row[0] for row in useful_info]
    # clean the id (want: o.1.1, but have: s1.1.1, o.col1.1.1)
    for index in range(len(id_col)):
        ans = id_col[index][0] # get only the first character
        numberings = re.findall(pattern = '[0-9]', string = id_col[index]) # get all numbers in the string
        for num in numberings:
            ans = ans + '.' + num
        id_col[index] = ans
    return id_col


def do_it_all(subcorpus_filepath):
    df = read_conll(subcorpus_filepath)
    # extract text number for later use
    text_number = re.findall(pattern = '.*\/([^\/]*)$', string = subcorpus_filepath)[0][0:-6]
    # get the ATF converted data's file path
    atf_converted_filepath = './atf_converted_data/' + text_number + '.conll'
    # check if this is a file
    if not os.path.isfile(atf_converted_filepath):
        print(text_number + ': This file errored out during the ATF conversion')
        return None

    full_id = extract_id(atf_converted_filepath)
    if len(full_id) != df.shape[0]:
        print(text_number + ': Rows of Subcorpus Data and ATF converted data do not match')
        return None
        
    # update the ID column
    df['ID'] = full_id
    # rename the WORD and MORPH column
    df = df.rename(columns={'WORD': 'FORM', 'MORPH': 'XPOSTAG'})
    # drop the EDGE and POS columns
    df = df.drop(columns = ['EDGE', 'POS'])
    # add the DEPREL column
    df['DEPREL'] = np.repeat('_', df.shape[0])
    # update the HEAD column
    df['HEAD'] = np.repeat('_', df.shape[0])
    # reorder the columns
    df = df[['ID', 'FORM', 'SEGM', 'XPOSTAG', 'HEAD', 'DEPREL', 'MISC']]

    with open(text_number + ".conll", "w") as fp:
        fp.write("#new_text=" + text_number + "\n")
    df.to_csv(text_number + '.conll', header = True, index = None, sep = '\t', mode = 'a')


######## get filepaths for files ########

subcorpus_directory = './royal_subcorpus_data'

filepaths = []
for filename in os.listdir(subcorpus_directory):
    f = os.path.join(subcorpus_directory, filename)
    # check if this is a file
    if os.path.isfile(f):
        filepaths += [str(f)]
filepaths.sort()

######## perform the read-clean-write for all files ########

for filepath in filepaths:
    do_it_all(filepath)