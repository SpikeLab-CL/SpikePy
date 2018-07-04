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
    def __init__(self, model, variables, encodings, explainer):
        self.model = model
        self.variables = variables
        self.encodings = encodings
        self.explainer = explainer
        self.load_encodings()
        self.load_variables()
        self.load_explainer()

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


    #TODO: handle missing value to pass to lime (try-except?)
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
            else:
                raw_value = arguments.get(col_name)
                if raw_value == "null":
                    vars_values[col_name] = np.nan
                    missing = True
                else:
                    vars_values[col_name] = float(raw_value)

        return np.array([vars_values[col] for col in self.variables])


        # Versión de matías
        """
         # arguments: ImmutableMultiDict([('var1', 'value1'), ('var2', 'value2')])
        instance = []
        for variable in self.variables:
            value = arguments.get(variable)
            if value is not None and value != "null"() and value != "":
                if variable in self.encodings:
                    if value not in self.encodings[variable].classes_:
                        instance.append(np.nan)  # si existe categoria nunca antes vista
                    else:
                        instance.append(self.encodings[variable].transform([value])[0])
                else:
                    instance.append(float(value))
            else:
                # si hay un missing lo reemplazamos con un nan en esa variable
                instance.append(np.nan)
        return instance
        """


    def predict_proba(self, this_array):
        h2o.init()
        shape_tuple = np.shape(this_array)
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)

        # Simplificar
        self.pandas_df = pd.DataFrame(data=this_array, columns=self.variables)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)

        self.aux_pred = self.model.predict(self.h2o_df)
        self.predictions = self.aux_pred.as_data_frame()
        self.predictions_result = self.predictions.iloc[:, 1:].as_matrix()

        h2o.remove(self.h2o_df.frame_id)  # remove frames to clean memory
        h2o.remove(self.aux_pred.frame_id)
        return self.predictions_result

    def get_explanation(self, instance):
        exp = self.explainer.explain_instance(instance,
                                              self.predict_proba,
                                              num_features=5)
        return exp

    def get_explanation_as_json(self, explanation, instance):
        explanation_as_list = explanation.as_list()
        json_response = []
        for variable in explanation_as_list:
            var = {}
            var['variable'] = variable[0]
            var['value'] = variable[1]
            json_response.append(var)

        probabilities = self.predict_proba(np.array(instance))[0]
        json_response.append({'Probability': [{'Autoriza': probabilities[0],
                                               'Rechaza-Redu': probabilities[1]}]})
        if probabilities[0] >= 0.8:
            json_response.append({'status': "Fast-Track"})
        elif probabilities[0] < 0.8 and probabilities[0] > 0.3:
            json_response.append({'status': "Sin recomendacion"})
        else:
            json_response.append({'status': 'Luz Roja'})
        return json.dumps(json_response)