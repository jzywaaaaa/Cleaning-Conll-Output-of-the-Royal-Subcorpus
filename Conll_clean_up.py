# author: Ziyue (Julia) Wang
# last updated: 2022 December 2

import pandas as pd
import re
import numpy as np
import sys

####### get files #######

subcorpus_file_loc = './royal_subcorpus_data/' + sys.argv[1] 
atf_converted_file_loc = './atf_converted_data/' + sys.argv[1]


######## define tools ########

# read_conll takes in a subcorpus conll file and outputs a dataframe
def read_conll(filepath):
    # read in the conll file
    data = pd.read_fwf(filepath)
    
    # deal with files with more than one line of comment in the beginning
    k = 0
    while data.iloc[k, 0][0] == '#':
        data = data.drop(k).reset_index(drop=True)
    
    # deal with files that automatically get its ID read into a separate column
    if '#' in data.columns: 
        data = data.drop(columns=['#'])

    # prepare a clean dataframe
    cleaned_data = pd.DataFrame({'WORD': [], 'SEGM': [], 'POS': [], \
        'MORPH': [], 'HEAD': [], 'EDGE': [], 'MISC': []})

    # extract useful information from the conll file
    useful_info = data.iloc[:, 0].str.split('\t')

    # update the clean dataframe
    for i in range(cleaned_data.shape[1]):
        cleaned_data.iloc[:, i] = [row[i] for row in useful_info]

    return cleaned_data


# extract_id takes an ATF converted file and extracts the ID column as a vector
def extract_id(filepath):
    # read in the conll file converted by the ATF converter
    data = pd.read_fwf(filepath)
    useful_info = data.iloc[1:, 0].str.split('\t')
    # get only the id
    id_col = [row[0] for row in useful_info]
    # clean the id (want: o.1.1, but have: s1.1.1, o.col1.1.1)
    for id in id_col:
        ans = id[0] # get only the first character
        numberings = re.findall(pattern = '[0-9]', string = id) # get all numbers in the string
        for num in numberings:
            ans = ans + '.' + num
        id = ans
    return id_col

def do_it_all(subcorpus_filepath, atf_converted_filepath):
    df = read_conll(subcorpus_filepath)
    # store the text number (e.g., P216736) for later use
    text_number = re.findall(pattern = '.*\/([^\/]*)$', string = subcorpus_filepath)[0][0:-6]
    
    # rename the WORD and MORPH column
    df = df.rename(columns={'WORD': 'FORM', 'MORPH': 'XPOSTAG'})
    # drop the EDGE and POS columns
    df = df.drop(columns = ['EDGE', 'POS'])
    # add the DEPREL column
    df['DEPREL'] = np.repeat('_', df.shape[0])
    # update the ID column
    df['ID'] = extract_id(atf_converted_filepath)
    # reorder the columns
    df = df[['ID', 'FORM', 'SEGM', 'XPOSTAG', 'HEAD', 'DEPREL', 'MISC']]

    with open(text_number + ".conll", "w") as fp:
        fp.write("#new_text=" + text_number + "\n")
    df.to_csv(text_number + '.conll', header = True, index = None, sep = '\t', mode = 'a')


######## perform the clean up ########

do_it_all(subcorpus_file_loc, atf_converted_file_loc)

