#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                 
# REVISED DATE: 
# PURPOSE: Create a function classify_images that uses the classifier function 
#          to create the classifier labels and then compares the classifier 
#          labels to the pet image labels. This function inputs:
#            -The Image Folder as image_dir within classify_images and function 
#             and as in_arg.dir for function call within main. 
#            -The results dictionary as results_dic within classify_images 
#             function and results for the functin call within main.
#            -The CNN model architecture as model wihtin classify_images function
#             and in_arg.arch for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           classifier label as the item at index 1 of the list and the comparison 
#           of the pet and classifier labels as the item at index 2 of the list.
#
##
# Imports classifier function for using CNN to classify images 
from classifier import classifier 

def classify_images(images_dir, results_dic, model):
    """
    Classifies images using a pre-trained model and updates the results dictionary with the model's predictions.

    Args:
    - images_dir: string, the directory containing the images to classify.
    - results_dic: dictionary, the dictionary containing the results to update.
    - model: string, the name of the pre-trained model to use for classification.

    Returns:
    - None
    """

    for key in results_dic:
        model_label = ""
        classified = classifier(images_dir+'/'+key, model)
 
        low_pet_image = classified.lower()

        # Split the classified label into a list of words
        word_list_pet_image = low_pet_image.split()

        # Remove non-alphabetic characters from each word in the list
        word_list_pet_image = [word for word in word_list_pet_image if word.isalpha()]

        # Join the list of words back into a single string to form the pet name
        pet_name = " ".join(word_list_pet_image)

        model_label = pet_name.strip()

        truth = results_dic[key][0]

        # Check if the truth label is in the model label
        if truth in model_label:
           results_dic[key].extend((model_label, 1))
        else:
           results_dic[key].extend((model_label, 0))

    print(results_dic)
