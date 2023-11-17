# -*- coding: utf-8 -*-
"""
src/utils/cross_validation.py
Created on: 25/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all the necessary plugins to cross-validate our models.
"""

# general imports
import numpy as np
from scipy import interp
import os
from math import sqrt

# matplotlib
import matplotlib.pyplot as plt

# static imports
from static.visualization import JAMA_COLORS, rgb_to_plt, OUT_STRINGS, MODEL_STRINGS
from static.files import OUTPUT_DIR

# sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold


def cross_validate_calibration(model: tuple,
                               data_dict: dict,
                               kfolds: int = 3,
                               fit: bool = True,
                               save_fig: bool = False,
                               show: bool = True,
                               ax: plt.Axes = None) -> None:
    """Method to perform calibration in a K-Fold cross validation setting.

    Parameters
    ----------
    model: tuple
        a tuple containing the name (index 0), features (2) and model itself (1)
    data_dict: dict
        the data dictionary containing the data
    kfolds: int
        the amount of K-Folds
    fit: bool
        boolean indicating whether or not to re-fit the classifier
    save_fig: bool
        boolean indicating whether or not to save the figure
    show: bool
        boolean indicating whether or not to show the figure
    ax: plt.Axes
        `matplotlib.Axes` object to plot the graph on

    Returns
    -------
    N/A

    """

    model_name = model[0]
    model = model[1]

    # Use axes if it is supplied
    if ax is not None:
        plt.sca(ax)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=kfolds)

    # Define empty lists
    frac_pos_list = []
    mean_pred_list = []
    briers_list = []

    i = 0
    x = data_dict['x']
    y = data_dict['y']

    # Loop through folds
    for train, test in cv.split(x, y):
        if fit:
            model.fit(x[train], y[train])

        probas_ = model.predict_proba(x[test])
        # Compute calibration curve
        frac_pos, mean_pred_val = calibration_curve(y[test], probas_[:, 1], n_bins=3)
        frac_pos_list.append(frac_pos)
        mean_pred_list.append(mean_pred_val)

        # Compute brier score
        brier_score = brier_score_loss(y[test], probas_[:, 1])
        briers_list.append(brier_score)

        # If show, plot individual calibration curves
        if show:
            plt.plot(mean_pred_val, frac_pos,
                     lw=1, alpha=0.3,
                     label='Fold %d (Brier = %0.2f)' % (i + 1, brier_score))

        i += 1

    # If showing, plot reference line
    if show:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Perfect calibration', alpha=.8)

    # Calculate statistics for Brier and fraction positives
    mean_brier = np.mean(briers_list)
    sd_brier = np.std(briers_list)
    mean_preds = np.mean(mean_pred_list, axis=0)
    mean_fracs = np.mean(frac_pos_list, axis=0)

    # Calculate confidence intervals for brier scores
    lowerci_brier = mean_brier - 1.96 * (sd_brier / sqrt(5))
    higherci_brier = mean_brier + 1.96 * (sd_brier / sqrt(5))

    # Print statistics
    print('Model: {}'.format(model))
    print('Brier: {:.3f} ({:.3f}-{:.3f})'.format(mean_brier, lowerci_brier, higherci_brier))

    if show:
        plt.plot(mean_preds, mean_fracs, color='b',
                 label=r'Mean = %0.2f (%0.2f - %0.2f)' % (mean_brier, lowerci_brier, higherci_brier),
                 lw=2, alpha=.8)

    if show:
        plt.ylabel('Fraction positives', fontsize=12)
        plt.xlabel('Mean predicted values', fontsize=12)
        plt.ylim([0, 1.0])
        plt.xlim([0, 1.0])
        plt.legend(loc="lower right",
                   frameon=True,
                   fontsize=8)
        plt.grid(linestyle=':')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    if save_fig:
        plt.savefig('CALIBRATION_{}_kfolds_{}.svg'.format(model_name, kfolds),
                    format='svg', dpi=1200)

    plt.show()


def cross_validate_ROC(model: list,
                       data_dict: dict,
                       kfolds: int = 3,
                       fit: bool = True,
                       save_fig: bool = False,
                       show: bool = True,
                       ax: plt.Axes = None) -> list:
    """Method to perform receiver operating characteristics (ROC) in a K-Fold cross validation setting.

    Parameters
    ----------
    model: tuple
        a tuple containing the name (index 0), features (2) and model itself (1)
    data_dict: dict
        the data dictionary containing the data
    kfolds: int
        the amount of K-Folds
    fit: bool
        boolean indicating whether or not to re-fit the classifier
    save_fig: bool
        boolean indicating whether or not to save the figure
    show: bool
        boolean indicating whether or not to show the figure
    ax: plt.Axes
        `matplotlib.Axes` object to plot the graph on

    Returns
    -------
    list
        a list containing:
            - true positive rates (TPR)
            - list of AUC (AUCs)
            - mean TPR
            - mean AUC
            - standard deviation of AUC
            - the plot object
    """

    model_name = model[0]
    model = model[1]
    
    if ax is not None:
        plt.sca(ax)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=kfolds)

    # Initialize empty lists
    tprs = []
    aucs = []
    accus = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    x = data_dict['x']
    y = data_dict['y']
    
    for train, test in cv.split(x, y):
        if fit:
            model.fit(x[train], y[train])
            
        probas_ = model.predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        accus.append(accuracy_score(y[test], probas_[:, 1] > 0.5)) 
        if show:
            plt.plot(fpr,
                     tpr,
                     lw=1,
                     alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i += 1
           
    if show:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Calculate statistics
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_accu = np.mean(accus)
    std_auc = np.std(aucs)
    std_accu = np.std(accus)

    # Calculate lower and higher confidence intervals for AUC and accuracy
    lower_ciauc = mean_auc - 1.96 * (std_auc / sqrt(5))
    higher_ciauc = mean_auc + 1.96 * (std_auc / sqrt(5))
    lower_ciaccu = mean_accu - 1.96 * (std_accu / sqrt(5))
    higher_ciaccu = mean_accu + 1.96 * (std_accu / sqrt(5))
    
    print('Model: {}'.format(model))
    print('AUC: {:.3f} ({:.3f}-{:.3f})'.format(mean_auc, lower_ciauc, higher_ciauc))
    print('ACCU: {:.3f} ({:.3f}-{:.3f})'.format(mean_accu, lower_ciaccu, higher_ciaccu))

    if show:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'AUC = %0.2f (%0.2f - %0.2f)' % (mean_auc, lower_ciauc, higher_ciauc),
                 lw=2, alpha=.8)

    # Calculate SD and ranges of true positive rates
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + (std_tpr / sqrt(4)) * 1.96, 1)
    tprs_lower = np.maximum(mean_tpr - (std_tpr / sqrt(4)) * 1.96, 0)

    if show:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 95% CI.')

    if show:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('1 - specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.legend(loc="lower right",
                   frameon=True,
                   fontsize=8)
        plt.grid(linestyle=':')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    # If save, save figure
    if save_fig:
        plt.savefig('model_{}_kfolds_{}.svg'.format(model_name, kfolds),
                    format='svg', dpi=1200)

    plt.show()
    
    return (tprs, aucs, mean_tpr, mean_auc, std_auc, plt)


def compare_models(models_list,
                   scores_list,
                   risk_score: bool = False,
                   save_fig: bool = False,
                   ax: plt.Axes = None) -> plt:
    """Method to perform comparison of the receiver operating characteristics (ROC) that
    were produced in a K-Fold cross validation setting.

    Parameters
    ----------
    models_list: list
        a list with the different models that were produced by utils.cross_validation.cross_validate_ROC
    scores_list: list
        a list with the different risk scores
    risk_score: bool
        boolean indicating whether or not we used a risk score for comparison
    save_fig: bool
        boolean indicating whether or not to save the figure
    ax: plt.Axes
        `matplotlib.Axes` object to plot the graph on

    Returns
    -------
    plt: plt
        returns the matplotlib object
    """

    # Define FPR space
    mean_fpr = np.linspace(0, 1, 100)

    # Init figure
    plt.figure()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(linestyle=':')

    # If ax is supplied, use it
    if ax is not None:
        plt.sca(ax)
    
    # Loop through models list
    for idx, values in enumerate(models_list):
        # extract model values
        name, model, out = values[0:3]
        
        # result from cross_validate_ROC()
        # last value is the plot value
        tprs, aucs, mean_tpr, mean_auc, std_auc, _ = values[3]
        plt.plot(mean_fpr, mean_tpr, color=rgb_to_plt(JAMA_COLORS[idx][0]),
                 label=r'%s - %0.2f (%0.2f - %0.2f)' % (MODEL_STRINGS[model], mean_auc,
                                                        mean_auc - (std_auc / sqrt(5)) * 1.96,
                                                        mean_auc + (std_auc / sqrt(5)) * 1.96),
                 lw=2, alpha=.8)
        
        # calculate std and create area around the mean ROC 
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + (std_tpr/sqrt(4)) * 1.96, 1)
        tprs_lower = np.maximum(mean_tpr - (std_tpr/sqrt(4)) * 1.96, 0)
        
        # Fill between the TPRs
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=rgb_to_plt(JAMA_COLORS[idx][1]), alpha=.3,)
                         #label=r'Model {} $\pm$ 1 std. dev.'.format(model))
    
    # if we require risk scores
    if risk_score:
        # calculate len scores list
        len_scores = len(models_list)
        
        # loop through risk scores
        for idx, values in enumerate(scores_list):
            # first is the name
            name = values[0]
            # second again are the values of the risk scores including the plt
            tprs, aucs, mean_tpr, mean_auc, std_auc = values[1]
            
            # plot
            plt.plot(mean_fpr, mean_tpr, color=rgb_to_plt(JAMA_COLORS[idx + len_scores][0]),
                     label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc),
                     lw=2, alpha=.8)
            
            # again fill between +- STD
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                             color=rgb_to_plt(JAMA_COLORS[idx + len_scores][1]),
                             alpha=.6,)
                             #label=r'{} $\pm$ 1 std. dev.'.format(name))

    # Further construct the plot
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('1 - specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    name = models_list[0][0]
    output = OUT_STRINGS[models_list[0][2]]

    plt.legend(loc="lower right",
               frameon=True,
               fontsize=8)

    plt.grid(linestyle=':')
    plt.gca().spines['top'].set_visible(False)
    
    # if we want to save individual plot
    if save_fig:
        plt.savefig(os.path.join(OUTPUT_DIR['ALGO_COMPARISON'], name.lower(),
                                 'model_comparison_output_{}.svg'.format(output.lower())),
                    format='svg', dpi=1200)
    
    # Show the plot
    plt.show()
        
    # return the plt
    return plt


def compare_algorithms(models_list: list,
                       save_fig: bool = False,
                       ax: plt.Axes = None) -> plt:
    """Method to perform comparison of the different algorithms.

    Parameters
    ----------
    models_list: list
        a list with the different models that were produced by utils.cross_validation.cross_validate_ROC
    save_fig: bool
        boolean indicating whether or not to save the figure
    ax: plt.Axes
        `matplotlib.Axes` object to plot the graph on

    Returns
    -------
    plt: plt
        returns the matplotlib object
    """

    # this is the space we want to detect it at
    mean_fpr = np.linspace(0, 1, 100)
    if ax is not None:
        plt.sca(ax)
    
    for idx, values in enumerate(models_list):
        # extract model values
        name, model, out = values[0:3]
        
        # result from cross_validate_ROC()
        # last value is the plot value
        tprs, aucs, mean_tpr, mean_auc, std_auc, _ = values[3]
        plt.plot(mean_fpr, mean_tpr, color=rgb_to_plt(JAMA_COLORS[idx][0]),
                 label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (model, mean_auc, std_auc),
                 lw=2, alpha=.8)
        
        # calculate std and create area around the mean ROC 
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # fill it
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=rgb_to_plt(JAMA_COLORS[idx][1]), alpha=.6,)
                         #label=r'Model {} $\pm$ 1 std. dev.'.format(model))
          
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    name = models_list[0][0]
    output = OUT_STRINGS[models_list[0][2]]
    
    #plt.title('{} prediction of {}'.format(name, output))
    plt.legend(loc="lower right",
               frameon=False, 
               fontsize=12)
    
    # if we want to save individual plot
    if save_fig:
        plt.savefig(os.path.join(OUTPUT_DIR['ALGO_COMPARISON'], name.lower(),
                                 'model_comparison_output_{}.svg'.format(output.lower())),
                    format='svg', dpi=1200)
    
    plt.show()
        
    # return the plt
    return plt


def cross_validate_risk_ROC(score: str,
                            data_dict: dict,
                            kfolds: int = 3,
                            save_fig: bool = False,
                            show: bool = True,
                            ax: plt.Axes = None) -> list:
    """Method to perform receiver operating characteristics (ROC) for
    the clinical risk scores in a K-Fold cross validation setting.

    Parameters
    ----------
    score: str
        string containing the cross-validated RISK score
    data_dict: dict
        the data dictionary containing the data
    kfolds: int
        the amount of K-Folds
    save_fig: bool
        boolean indicating whether or not to save the figure
    show: bool
        boolean indicating whether or not to show the figure
    ax: plt.Axes
        `matplotlib.Axes` object to plot the graph on

    Returns
    -------
    list
        a list containing:
            - true positive rates (TPR)
            - list of AUC (AUCs)
            - mean TPR
            - mean AUC
            - standard deviation of AUC
    """

    if ax is not None:
        plt.sca(ax)
        
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=kfolds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    x = data_dict['x']
    y = data_dict['y']
    
    for train, test in cv.split(x, y):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[train], x[train])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if show:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
        
    if show:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if show:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if show:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

    if show: 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic - {}'.format(score))
        plt.legend(loc="lower right",
                   frameon=False, 
                   fontsize=8)
    if save_fig:
        plt.savefig(os.path.join(OUTPUT_DIR['ALGO_COMPARISON'], score.lower(),
                                 'score_{}_kfolds_{}.svg'.format(score, kfolds)),
                    format='svg', dpi=1200)
    if show:
        plt.show()
    
    return tprs, aucs, mean_tpr, mean_auc, std_auc
