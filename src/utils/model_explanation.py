# -*- coding: utf-8 -*-
"""
src/utils/model_explanation.py
Created on: 26/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all the necessary plugins to explain our models.
"""

# general imports
import shap
import pandas as pd
import os

# static import
from src.static.files import OUTPUT_DIR

# matplotlib
import matplotlib.pyplot as plt


def generate_individual_shap_predictions(model: object,
                                         data_dict: dict,
                                         to_pdf: bool = False) -> None:
    """Method to generate individual SHAP predictions.

    Parameters
    ----------
    model: object
        the model instance object
    data_dict: dict
        the data dictionary
    to_pdf: bool
        if True, will export results to a PDF file

    Returns
    -------
    N/A

    """
    
    # generate explainer model 
    explainer = shap.TreeExplainer(model)
    
    # obtain shap values for this specific data dict
    shap_values = explainer.shap_values(data_dict['x'])
    
    # generate an internal dataframe with the features
    internal_df = pd.DataFrame(data_dict['x'], columns=data_dict['features'])
    
    # if output is pdf!
    if to_pdf:
        # import pdf backend
        import matplotlib.backends.backend_pdf
        
        # create pfd document
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(OUTPUT_DIR['MODEL_EXPLANATION'],
                                                                    'output.pdf'))
        # loop through the data_dict, row for row
        for index in range(data_dict['x'].shape[0]):
            # create a force-plot
            shap.force_plot(explainer.expected_value, shap_values[index, :], internal_df.iloc[index],
                            matplotlib=True, show=False, link='logit', text_rotation=45)
            plt.gcf().set_size_inches(11.69, 8.27)
            plt.gcf().tight_layout()
            # save PDF figure
            pdf.savefig()
        
        # close pdf
        pdf.close()
    
    # if output is directly to the interface
    else:
        # loop through the data_dict, row for row
        for index in range(data_dict['x'].shape[0]):
            # make a force plot and show it!
            shap.force_plot(explainer.expected_value, shap_values[index, :], internal_df.iloc[index],
                            matplotlib=True, show=True, link='logit')
