"""src/utils/files.py
Created on: ??
Last edited: 17/07/2020
Author: William van Doorn

This utils script contains all the utils to be
used with file handling, model saving, etc. It
only consists of public functions and no other classes.

Methods
-------
    load_pickle(file)
        Loads a pickle on the specified file path
    save_pickle(obj, file)
        Pickles an object and saves it to the file path
    save_model(file, model)
    
    load_model(file, type_model)
"""

# general imports
import os, glob

# logging
import logging, traceback
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def save_model(file: str,
               model: object) -> None:
    """Saves a specific model to the specified
    file path. Throws exception if saving is not 
    successful.

    Parameters
    ----------
    file: str
        the path to the file which should be saved.
    model: object
        the specific model instance to be saved
            

    Returns
    -------
    N/A.
    
    """  
    
    from keras.models import Model
    import pickle
    from xgboost import XGBClassifier
    try:
        if isinstance(model, XGBClassifier):
            model.save_model(os.path.join('models',
                                          'xgboost2',
                                          file))

        elif isinstance(model, Model):
            model.save(os.path.join('models',
                                    'keras',
                                    file))
        else:
            pickle.dump(model, 
                        open(os.path.join('models',
                                          'others',
                                          file), 'wb'))
            
        logging.info('Model {} successfully saved'.format(model))
    except Exception as e:
        logging.error("Failed saving model {}".format(model))
        logging.error(traceback.format_exc()) 


def load_model(file: str,
               type_model: str) -> object:
    """Loads a specific model from the specified
    file path. Throws exception if saving is not 
    successful.

    Parameters
    ----------
    file: str
        the path to the file which should be loaded.
    type_model: str
        string containing the type of model, see source code
        documentation for full details.
            
    Returns
    -------
    sklearn.models.Model/xgboost2.XGBClassifier
        returns model instance according to the `type_model` parameter
    
    """
    
    from keras.models import load_model
    import pickle
    from xgboost import Booster
    
    try:
        if type_model == 'xgboost2':
            return Booster().load_model(os.path.join('models',
                                                     'xgboost2',
                                                     file))

        elif type_model == 'keras':
            return load_model(os.path.join('models',
                                           'keras',
                                           file))
        else:
            return pickle.load(open(os.path.join('models',
                                                 'others',
                                                 type_model), 'rb'))
    
    except Exception as e:
        logging.error("Failed loading model {}".format(file))
        logging.error(traceback.format_exc()) 
        return None


def load_pickle(file: str) -> object:
    """Loads a pickle file from supplied path.
    Returns an logging error and None object otherwise.

    Parameters
    ----------
    file: str
        The path to the file which should be loaded.
            

    Returns
    -------
    pickle
        Returns pickled object if existing, otherwise None. 
    """  
    
    import pickle
    try:
        return pickle.load(open(file, "rb"))
    except (OSError, IOError) as e:
        logging.error("Pickle file not found!")
        return None


def save_pickle(obj: object, file: str) -> object:
    """Saves a pickle file to the supplied path.
    Returns an logging error and None object otherwise.

    Parameters
    ----------
    obj : object
        The object to be pickled.
    file : str
        The path to the file which should be loaded.

    Returns
    -------
    object
        Returns pickled object if successful, otherwise None.
    """  
    
    import pickle
    try:
        return pickle.dump(obj, open(file, "wb"))
    except (OSError, IOError) as e:
        logging.error("Not able to save a pickle file!")
        return None


def get_latest_file(path: str, extension: str = '*.csv') -> str:
    """Returns the latest file in a specific directory
    with a specific extension.

    Parameters
    ----------
    path: str
        The path to the file which should be loaded.
    extension: str
        The extension of the file to search for.

    Returns
    -------
    str
        Returns the latest file
    """

    # list files
    list_of_files = glob.iglob('{}/{}'.format(path,
                                              extension))
    # return max() based on getctime
    return max(list_of_files, key=os.path.getctime)


def get_latest_version(path: str, extension: str = '*') -> int:
    """Returns the latest version of a file in a specific directory
    with a specific extension.

    Parameters
    ----------
    path: str
        The path to the file which should be loaded.
    extension: str
        The extension of the file to search for.

    Returns
    -------
    int
        Returns the latest version
    """

    try:
        return int(get_latest_file(path, extension).
                   split('\\')[-1].split('-')[0].split('v')[1])
    except Exception as e:
        return 0


def create_directory(path: str) -> None:
    """Create a directory at the specified path.

    Parameters
    ----------
    path: str
        The path to the file which should be loaded.

    Returns
    -------
    N/A.
    """

    try:  
        os.mkdir(path)
    except OSError:  
        logging.error("Creation of the directory {} failed".format(path))
    else:  
        logging.info("Successfully created the directory: {}".format(path))
