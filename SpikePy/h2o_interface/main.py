###########
#
#
# Introduces different function to interface to h2o in an easier way
# it also extends h2o to use with LIME
#
# March, 2018
#

import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import LabelEncoder
import h2o
from lime.lime_tabular import LimeTabularExplainer


# 1. Lime extension for in memory model
#
# Pasos
#
# 1. Preparar datos (label encoding)
# 2. Pasar a H2OFrame
# 3. Estimar modelo y obtener interpretaciones de Lime
########################################


# 1. Preparar datos (label encoding)
########################################

class LimeDF:
    """
    Stores info about a pandas dataframe, including labels
    Has a method for producing a numpy array to be passed to LIME

    df : pd.DataFrame
        Full dataset with features and target
    x_vars : Union[list, np.array]
        List of all X features, including numerical and categorical
    categorical_cols: Union[list, np.array]
        List of names of categorical columns
    y_var : string
        Name Y variable (aka target)
    """

    def __init__(self, df: pd.DataFrame, x_vars: Union[list, np.array],
                 categorical_cols: Union[list, np.array], y_var: str):
        df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('object'))
        self.df = df
        self.categorical_cols = categorical_cols
        self.y_var = y_var
        self.x_vars = x_vars
        self.categorical_cols_ind = ([x_vars.index(categorical_cols[n])
                                      for n in range(len(categorical_cols))])

        # Place-holders
        self.y_class_names = None
        self.categorical_names_dict = None
        self.y_labels = None

    def to_numpy_array(self):
        # Handle Y
        le = LabelEncoder()
        self.y_labels = le.fit_transform(self.df[self.y_var])
        self.y_class_names = le.classes_

        # Handle categorical Xs
        data = self.df[self.x_vars].as_matrix()
        self.categorical_names_dict = {}
        for feature_ind in self.categorical_cols_ind:
            print("Doing: ", self.x_vars[feature_ind], "with id ", feature_ind)
            le = LabelEncoder()
            data[:, feature_ind] = le.fit_transform(data[:, feature_ind].astype(str))
            self.categorical_names_dict[feature_ind] = le.classes_

        return data.astype(float)

    def from_h2o_to_numpy_array(self, h2o_frame):
        """
        Converts an h2o frame to a numpy array to be used
        by the explainer object

        :param h2o_frame: usually train or test object
        :return: np.array
        """
        return h2o_frame[self.x_vars].as_data_frame().as_matrix()

    def tabular_explainer(self, h2o_train_frame, **kwargs):
        """

        :param h2o_train_frame:
        :param kwargs: kernel_width, verbose, etc
        :return: LimeTabularExplainer object
        """
        return LimeTabularExplainer(self.from_h2o_to_numpy_array(h2o_train_frame),
                                    feature_names=self.x_vars,
                                    class_names=self.y_class_names,
                                    categorical_features=self.categorical_cols_ind,
                                    categorical_names=self.categorical_names_dict,
                                    **kwargs)


class H2oPredictProbaWrapper:
    # drf is the h2o distributed random forest object, the column_names is the
    # labels of the X values
    def __init__(self, model, column_names):
        self.model = model
        self.column_names = column_names

    def predict_proba(self, this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)

        # We convert the numpy array that Lime sends to a pandas dataframe and
        # convert the pandas dataframe to an h2o frame
        self.pandas_df = pd.DataFrame(data=this_array, columns=self.column_names)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)

        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame()
        # the first column is the class labels, the rest are probabilities for
        # each class
        self.predictions = self.predictions.iloc[:, 1:].as_matrix()
        return self.predictions
