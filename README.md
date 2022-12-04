# Cleaning Conll Output of the Royal Subcorpus

## Goal

This tool cleans the [non-standard conll data of the Royal Subcorpus](https://github.com/cdli-gh/mtaac_syntax_corpus/tree/master/royal/release) to cdli-conll format. 

#### Example: P216726.conll

The non-standard conll data of P216726 looks like this:

````
# global.columns = ID WORD SEGM POS MORPH HEAD EDGE MISC
1	da-da	da-da[3]	PN	PN	11	ERG	_
2	ensi2	ensi2[ruler]	N	N	1	appos	_
3	szuruppak{ki}	szuruppak{ki}[1]	SN	SN.GEN	2	GEN	_
4	ha-la-ad-da	ha-la-ad-da[1]	PN	PN	1	appos	_
5	ensi2	ensi2[ruler]	N	N	1	appos	_
6	szuruppak{ki}	szuruppak{ki}[1]	SN	SN.GEN	5	GEN	_
7	dumu-ni	dumu[child]	N	N.3-SG-H-POSS.ERG	1	appos	_
8	ad-us2	ad-us2[plank]	N	N.ABS	11	ABS	_
9	abul	abul[gate]	N	N	11	LOC	_
10	{d}sud3-da-ke4	{d}sud3[1]	DN	DN.GEN.L3-NH	9	GEN	_
11	bi2-in-us2	us2[follow]	V	3-NH.L3.3-SG-H-A.V.3-SG-P	0	_	_
````
Whereas the gold-standard cdli-conll data (after the clean up) looks like this:
````
#new_text=P216736
ID	FORM	SEGM	XPOSTAG	HEAD	DEPREL	MISC
a.1.1	da-da	da-da[3]	PN	11	_	_
a.2.1	ensi2	ensi2[ruler]	N	1	_	_
a.3.1	szuruppak{ki}	szuruppak{ki}[1]	SN.GEN	2	_	_
a.4.1	ha-la-ad-da	ha-la-ad-da[1]	PN	1	_	_
a.5.1	ensi2	ensi2[ruler]	N	1	_	_
a.6.1	szuruppak{ki}	szuruppak{ki}[1]	SN.GEN	5	_	_
a.7.1	dumu-ni	dumu[child]	N.3-SG-H-POSS.ERG	1	_	_
a.8.1	ad-us2	ad-us2[plank]	N.ABS	11	_	_
a.8.2	abul	abul[gate]	N	11	_	_
a.9.1	{d}sud3-da-ke4	{d}sud3[1]	DN.GEN.L3-NH	9	_	_
a.1.0.1	bi2-in-us2	us2[follow]	3-NH.L3.3-SG-H-A.V.3-SG-P	0	_	_
````

More examples of gold-standard cdli-conll data can be found [here](https://github.com/cdli-gh/mtaac_gold_corpus/tree/workflow/morph/to_dict).

## Usage

1. Download the non-standard conll data of the Royal Subcorpus either from its [original repo](https://github.com/cdli-gh/mtaac_syntax_corpus/tree/master/royal/release/data) or from the zip file provided in this repo. You MUST name the folder __royal_subcorpus_data__.

2. Download the conll files generated by the ATF converter:
    1. Download the original data dump from [here](https://github.com/cdli-gh/data/blob/master/cdliatf_unblocked.atf);
    2. Install and then run the ATF converter on the data dump according to [this repo](https://github.com/cdli-gh/atf2conll-convertor);
    3. Remember: you MUST name the resulting folder __royal_subcorpus_data__.

3. Download the Conll_clean_up.py. Make sure that the two folers (royal_subcorpus_data and atf_converted_data) and the python script are in the same working directory.

4. Open a terminal at the folder and run the following command:
````
python3 Conll_clean_up.py
````
You should then see the cleaned up gold-standard conll files in the folder. Any files erroring out in the clean-up process should be indicated by a message printed out.


## Caution

1. The two sources of data must be names __royal_subcorpus_data__ and __royal_subcorpus_data__ respectively and put in the same working directory as Conll_clean_up.py.
2. If you want to re-run the clean up on the folder, please remove the previously generated files from your working directory first.
3. Conll_clean_up.py is written with the assumption that the user wants to clean up a folder of files. If you only want to clean up one non-standard royal subcorpus conll file, please open the python script and use the `do_it_all(filepath)` function in it and let the input be the file path of the non-standard royal subcorpus conll file you want to clean.
