import h2o
# 1. Load data
############


# 2. Pasar a h2o
########################################

h2o.init(nthreads=-1, max_mem_size='2000M')
h2o.remove_all()

# train
train_h2o_df = h2o.H2OFrame(train)
train_h2o_df.set_names(features)
train_h2o_df['resolucion'] = h2o.H2OFrame(labels_train)
train_h2o_df['resolucion'] = train_h2o_df['resolucion'].asfactor()

# test
test_h2o_df = h2o.H2OFrame(test)
test_h2o_df.set_names(features)
test_h2o_df['resolucion'] = h2o.H2OFrame(labels_test)
test_h2o_df['resolucion'] = test_h2o_df['resolucion'].asfactor()

# Explícitamente transformar a categóricas
for feature in categorical_features:
    train_h2o_df[feature] = train_h2o_df[feature].asfactor()
    test_h2o_df[feature] = test_h2o_df[feature].asfactor()

# 3. Estimar modelo y obtener explicaciones LIME!
################################################

gbm1 = H2OGradientBoostingEstimator(
    model_id='gbm_v1',
    seed=1234,
    ntrees=8000,
    nfolds=5,
    stopping_metric='mse',
    stopping_rounds=15,
    score_tree_interval=3)

gbm1.train(x=features,
           y='resolucion', training_frame=train_h2o_df)

gbm1.model_performance(test_h2o_df)


class h2o_predict_proba_wrapper:
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


h2o_drf_wrapper = h2o_predict_proba_wrapper(gbm1, features)

explainer = LimeTabularExplainer(train, feature_names=features, class_names=class_names,
                                 categorical_features=categorical_features,
                                 categorical_names=categorical_names,
                                 kernel_width=None,
                                 verbose=True)


