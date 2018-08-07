import dill
import h2o
import numpy as np
import pandas as pd
import json


class ProductionModel(object):
    """
    Contains h2o model, variable list and encodings. Ready
    to be used in a flask or dash app.

    """
    def __init__(self, model, variables, encodings, explainer, column_types, mode):
        self.model = model
        self.variables = variables
        self.encodings = encodings
        self.explainer = explainer
        # self.load_encodings()
        # self.load_variables()
        self.load_explainer()

        self.column_types = column_types
        self.mode = mode

    def load_encodings(self):
        try:
            with open(self.encodings, 'rb') as f:
                self.encodings = dill.load(f)
        except IOError as e:
            print("Can't find file: {0}. Error: {1}".format(self.encodings, e))

    def load_variables(self):
        try:
            with open(self.variables, 'rb') as f:
                self.variables = dill.load(f)
        except IOError as e:
            print("Can't find file: {0}. Error: {1}".format(self.encodings, e))

    @property
    def rendered_variables(self):
        variables = []
        for variable in self.variables:
            variables.append({
                "name": variable,
                "classes": list(self.encodings[variable].classes_)
                if variable in self.encodings else None
                })
        return variables

    def load_explainer(self):
        try:
            with open(self.explainer, 'rb') as f:
                self.explainer = dill.load(f)
        except IOError as e:
            print("Can't find explainer at: {0}. Error: {1}".format(self.explainer, e))


    #TODO: handle missing value in a more elegant way?
    def process_input(self, arguments):
        missing = False

        # Float vs categ columns
        vars_values = {}
        for col_name in self.variables:
            if col_name in self.encodings:
                non_encoded = arguments.get(col_name)
                # null de categórica o categoría nunca antes vista
                if non_encoded not in self.encodings[col_name].classes_:
                    vars_values[col_name] = np.nan
                    missing = True
                else:
                    vars_values[col_name] = self.encodings[col_name].transform([non_encoded])
            else:  # float case
                raw_value = arguments.get(col_name)
                if raw_value == "null":
                    vars_values[col_name] = np.nan
                    missing = True
                else:
                    vars_values[col_name] = float(raw_value)

        return missing, np.array([vars_values[col] for col in self.variables])

    def predict_proba(self, this_array):
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)
        one_observation = False
        if len(shape_tuple) == 1:
            one_observation = True
            this_array = this_array.reshape(1, -1)

        # Manage missing values:
        h2o_df = h2o.H2OFrame(this_array, column_names=self.variables,
                              column_types=self.column_types, destination_frame="scratch")

        predictions = self.model.predict(h2o_df).as_data_frame()

        if self.mode == "classification":
            # first column is class labels, the rest
            # are probabilities for each class
            predictions = predictions.iloc[:, 1:].as_matrix()
        elif self.mode == "regression":
            if one_observation:
                predictions = predictions.values[0]
            else:
                # TODO this still doesn't work
                predictions = predictions.values[:, 0]
        else:
            raise AttributeError("Mode must be either classification or regression")

        return predictions

    def get_explanation(self, instance):
        exp = self.explainer.explain_instance(instance,
                                              self.predict_proba,
                                              num_features=5)
        return exp

    @staticmethod
    def fast_track_status(probability):
        if probability >= 0.8:
            status = "Fast-Track"
        elif 0.3 < probability < 0.8:
            status = "Sin recomendacion"
        else:
            status = "Luz Roja"
        return {'status': status}



    def get_explanation_as_json(self, explanation, instance):
        explanation_as_list = explanation.as_list()
        json_response = []
        for variable in explanation_as_list:
            var = dict()
            var['variable'] = variable[0]
            var['value'] = variable[1]
            json_response.append(var)

        probabilities = self.predict_proba(np.array(instance))[0]
        json_response.append({'Probability': [{'Autoriza': probabilities[0],
                                               'Rechaza-Redu': probabilities[1]}]})
        probability = probabilities[0]
        json_response.append(ProductionModel.fast_track_status(probability))

        return json.dumps(json_response)