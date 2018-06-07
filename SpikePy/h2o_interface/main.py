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
import os
import os.path


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
        List of names of categorical columns. Pass empty list if there are no categorical columns.
    y_var : string
        Name Y variable (aka target)
    y_categorical: bool
        Whether Y is categorical or not
    label_encodings : dict
        Label encodings for categorical vars (for export). Doesn't handle new categories
    """

    def __init__(self, df: pd.DataFrame, x_vars: Union[list, np.array],
                 categorical_cols: Union[list, np.array], y_var: str,
                 y_categorical: bool, label_encodings=dict()):

        if len(categorical_cols) > 0:
            df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('object'))
        self.df = df
        self.categorical_cols = categorical_cols
        self.y_var = y_var
        self.x_vars = x_vars
        if len(categorical_cols) > 0:
            self.categorical_cols_ind = ([x_vars.index(categorical_cols[n])
                                          for n in range(len(categorical_cols))])
        else:
            self.categorical_cols_ind = None

        # Place-holders
        self.y_class_names = None
        self.categorical_names_dict = None
        self.y_labels = None
        self.y_categorical = y_categorical
        self.label_encodings = label_encodings

    def to_numpy_array(self):
        if self.y_categorical:
            le = LabelEncoder()
            self.y_labels = le.fit_transform(self.df[self.y_var])
            self.y_class_names = le.classes_
        else:
            # Y_labels is just the raw y_var
            self.y_labels = self.df[self.y_var].values

        data = self.df[self.x_vars].as_matrix()
               
        # Handle categorical Xs
        if len(self.categorical_cols) > 0:
            self.label_encodings = dict()
            self.categorical_names_dict = {}
            for feature_ind in self.categorical_cols_ind:
                print("Doing: ", self.x_vars[feature_ind], "with id ", feature_ind)
                le = LabelEncoder()
                data[:, feature_ind] = le.fit_transform(data[:, feature_ind].astype(str))
                self.categorical_names_dict[feature_ind] = le.classes_

                # Save label encoding
                col_name = self.x_vars[feature_ind]
                self.label_encodings[col_name] = le

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
        :type kwargs: dict
        :return: LimeTabularExplainer object
        """

        if self.y_categorical:
            class_names = self.y_class_names
            mode = "classification"
        else:
            class_names = [self.y_var]  # Not sure this one matters or not
            mode = "regression"

        return LimeTabularExplainer(self.from_h2o_to_numpy_array(h2o_train_frame),
                                    feature_names=self.x_vars,
                                    class_names=class_names,
                                    categorical_features=self.categorical_cols_ind,
                                    categorical_names=self.categorical_names_dict,
                                    mode=mode, **kwargs)


class H2oPredictProbaWrapper:
    """
    model : h2o model
    mode : "classification" or "regression"
    """

    def __init__(self, model, column_names, mode: str, column_types: dict):
        self.model = model
        self.column_names = column_names
        self.column_types = column_types
        self.mode = mode

        # Place-holders
        self.pandas_df = None
        self.h2o_df = None
        self.predictions = None

    def predict_proba(self, this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)
        one_observation = False
        if len(shape_tuple) == 1:
            one_observation = True
            this_array = this_array.reshape(1, -1)


        #self.pandas_df = pd.DataFrame(data=this_array, columns=self.column_names)
        #self.h2o_df = h2o.H2OFrame(self.pandas_df)

        # Manage missing values:
        self.h2o_df = h2o.H2OFrame(this_array, column_names=self.column_names,
                                column_types=self.column_types)

        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame()

        if self.mode == "classification":
            # the first column is the class labels, the rest are probabilities for
            # each class
            self.predictions = self.predictions.iloc[:, 1:].as_matrix()
        elif self.mode == "regression":
            if one_observation:
                self.predictions = self.predictions.values[0]
            else:
                # TODO this still doesn't work
                self.predictions = self.predictions.values[:, 0]
        else:
            raise AttributeError("Mode must be either classification or regression")

        return self.predictions


class H2oOutOfMemoryWrapper:
    """
    Wrapper para poder hacer predicciones a partir de un modelo de h2o (MOJO y h2o.genmodel.jar),
    funciona sin la necesidad de levantar h2o, para esto se debe disponer del siguiente directorio con
    los siguientes archivos:
        -H2OLimeWrapper.py
            -/input Carpeta con el documento de entrada
            -/output Carpeta con resultados de predicciones
            -/train_data Carpeta con el archivo que contiene los datos de entrenamiento
            -/models: Carpeta contenedora del MOJO y h2o.genmodel.jar

    Args:
        input_name (str): Nombre del archivo con los datos (solamente Xs) para hacer la prediccion.
        output_name (str): Nombre del archivo para guardar el resultado de las predicciones.
        train_data (str): Nombre del archivo con los datos (solamente Xs) con los que se entrenó el modelo.
        model_name (str): Nombre del modelo MOJO
        :label_encodings: dict . default None : Dictionary of label encodings for X and Y variables
    """

    def __init__(self, input_name, output_name, train_data, model_name, label_encodings=None):
        """
        :param input_name:
        :param output_name:
        :param train_data:
        :param model_name:
        :param label_encodings:
        :type label_encodings: dict
        """
        self.train_data = train_data
        self.input_name = input_name
        self.output_name = output_name
        self.model_name = model_name
        self.work_directory = os.getcwd()
        self.checkFiles()

        train_sample = pd.read_csv("{0}/train_data/{1}".format(self.work_directory, self.train_data),
                                   nrows=5)
        self.x_vars = train_sample.columns
        self.label_encodings = label_encodings

    def checkFiles(self):
        """
            Returns:
                None: Checkea que los archivos necesarios se encuentren en las carpetas correspondientes
        """
        if os.path.isfile("models/{0}".format(self.model_name)) is False:
            raise ValueError("No se encuentra el modelo {0}, \
                              en la carpeta models".format(self.model_name))
        if os.path.isfile("input/{0}".format(self.input_name)) is False:
            raise ValueError("No se encuentra el archivo {0}, \
                              en la carpeta models".format(self.model_name))
        if os.path.isfile("train_data/{0}".format(self.train_data)) is False:
            raise ValueError("No se encuentra el archivo {0}, \
                              en la carpeta train_data".format(self.train_data))

    def get_train_data(self):
        """
            Returns:
                pandas.DataFrame: Dataframe con los datos de entrenamiento del modelo
        """
        return pd.read_csv("{0}/train_data/{1}".format(self.work_directory, self.train_data))

    # TODO check if code changes if Y is float
    def make_predictions(self, input_name):
        """
            Args:
                input_name (str): Nombre del archivo con los datos a predecir
            Returns:
                pandas.DataFrame: Probabilidades de cada instancia
        """
        cmd = "java -cp {0}/models/h2o-genmodel.jar hex.genmodel.tools.PredictCsv  \
               --header --mojo {0}/models/{1} --input {0}/input/{3} \
               --output {0}/output/{2} --decimal".format(self.work_directory,
                                                         self.model_name,
                                                         self.output_name,
                                                         input_name)
        os.system(cmd)
        return self.load_results()

    def load_results(self):
        """
            Returns:
                pandas.DataFrame: Dataframe con los resultados de la prediccion de los datos de entrada
        """
        predictions = pd.read_csv("{0}/output/{1}".format(self.work_directory, self.output_name))
        return predictions

    def get_input_data(self, encode_categories=False):
        """
            Returns:
                pandas.DataFrame: Dataframe con los datos a predecir
        """
        raw_df = pd.read_csv("{0}/input/{1}".format(self.work_directory, self.input_name))
        if encode_categories:
            # Use label encodings for the categorical Xs
            for col_name, col_encoder in self.label_encodings.items():
                raw_df[col_name] = col_encoder.transform(raw_df[col_name].astype(str))

        return raw_df

    # TODO handle possibility of y being float or int
    def limePredictions(self, data):
        """
            Args:
                data: (np.array): Arreglo de datos a predecir
            Returns:
                np.array : Probabilidades de cada instancia generada por LIME
        """
        frame = pd.DataFrame(data=data, columns=self.x_vars)
        frame.to_csv("{0}/input/parcial_predictions.csv".format(self.work_directory), sep=',')
        pred = self.make_predictions(input_name="parcial_predictions.csv")

        shape_tuple = np.shape(pred)
        if len(shape_tuple) == 1:
            pred = pred.reshape(1, -1)

        pred = pred.iloc[:, 1:].as_matrix()
        pred[-1][1] = 1 - pred[-1][0]
        return pred
