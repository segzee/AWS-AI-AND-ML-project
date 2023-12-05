#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#                                                                             
# PROGRAMMER: segun
# DATE CREATED:                                 
# REVISED DATE: 
# PURPOSE: Create a function adjust_results4_isadog that adjusts the results 
#          dictionary to indicate whether or not the pet image label is of-a-dog, 
#          and to indicate whether or not the classifier image label is of-a-dog.
#          All dog labels from both the pet images and the classifier function
#          will be found in the dognames.txt file. We recommend reading all the
#          dog names in dognames.txt into a dictionary where the 'key' is the 
#          dog name (from dognames.txt) and the 'value' is one. If a label is 
#          found to exist within this dictionary of dog names then the label 
#          is of-a-dog, otherwise the label isn't of a dog. Alternatively one 
#          could also read all the dog names into a list and then if the label
#          is found to exist within this list - the label is of-a-dog, otherwise
#          the label isn't of a dog. 
#         This function inputs:
#            -The results dictionary as results_dic within adjust_results4_isadog 
#             function and results for the function call within main.
#            -The text file with dog names as dogfile within adjust_results4_isadog
#             function and in_arg.dogfile for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           whether or not the pet image label is of-a-dog as the item at index
#           3 of the list and whether or not the classifier label is of-a-dog as
#           the item at index 4 of the list. Note we recommend setting the values
#           at indices 3 & 4 to 1 when the label is of-a-dog and to 0 when the 
#           label isn't a dog.
#
##
# TODO 4: Define adjust_results4_isadog function below, specifically replace the None
#       below by the function definition of the adjust_results4_isadog function. 
#       Notice that this function doesn't return anything because the 
#       results_dic dictionary that is passed into the function is a mutable 
#       data type so no return is needed.
# 
def adjust_results4_isadog(results_dic, dogfile):
   
    # Creates dognames dictionary for quick matching to results_dic labels from
    # real answer & classifier's answer
    dognames_dic = {}

    # Reads in dognames from file, 1 name per line & automatically closes file
    with open(dogfile, "r") as infile:
        # Reads in dognames from first line in file
        line = infile.readline()

        # Processes each line in file until reaching EOF (end-of-file) by 
        # processing line and adding dognames to dognames_dic with while loop
        while line != "":
            # Strip newline character from line
            line = line.strip()

            # Adds dogname to dognames_dic if it doesn't already exist in the dictionary
            if line not in dognames_dic:
                dognames_dic[line] = 1

            # Reads in next line in file to be processed with while loop
            # if this line isn't empty (EOF)
            line = infile.readline()

    # Add to whether pet labels & classifier labels are dogs by appending
    # two items to end of value(List) in results_dic. 
    # List Index 3 = whether(1) or not(0) Pet Image Label is a dog AND 
    # List Index 4 = whether(1) or not(0) Classifier Label is a dog
    # How - iterate through results_dic if labels are found in dognames_dic
    # then label "is a dog" index3/4=1 otherwise index3/4=0 "not a dog"
    for key in results_dic:
        if len(results_dic[key]) >= 3:

        # Pet Image Label IS of Dog (e.g. found in dognames_dic)
            if results_dic[key][0] in dognames_dic:
            
            # Classifier Label IS image of Dog (e.g. found in dognames_dic)
            # appends (1, 1) because both labels are dogs
                if results_dic[key][1] in dognames_dic:
                    results_dic[key].extend((1, 1))

            # Classifier Label IS NOT image of dog (e.g. NOT in dognames_dic)
            # appends (1,0) because only pet label is a dog
                else:
                    results_dic[key].extend((1, 0))

        # Pet Image Label IS NOT a Dog image (e.g. NOT found in dognames_dic)
            else:
            
            # Classifier Label IS image of Dog (e.g. found in dognames_dic)
            # appends (0, 1)because only Classifier label is a dog
                if results_dic[key][1] in dognames_dic:
                    results_dic[key].extend((0, 1))

            # Classifier Label IS NOT image of Dog (e.g. NOT in dognames_dic)
            # appends (0, 0) because both labels aren't dogs
                else:
                    results_dic[key].extend((0, 0))
                
    return results_dic