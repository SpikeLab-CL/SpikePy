import dill


def export_flask_dict(limedf, dfh2o, location="", model_name=""):
    """
    Takes a LimeDF object and the h2o dataframe with training or test data
    and exports the flask dictionary necessary for using flask with the model
    """
    flask_d = {'encodings': limedf.label_encodings,
               'column_types': dfh2o[limedf.x_vars].types,
               'column_names': limedf.x_vars,
               'version': model_name}

    with open(location + "dict_for_flask.dill", 'wb') as f:
        dill.dump(flask_d, f)
        print("Dictionary saved at ", location + "dict_for_flask.dill")
