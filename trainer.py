
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from sklearn.pipeline import make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

N = 10000
df = get_data(nrows=N)
df = clean_data(df)
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class Trainer():
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())

        # column transformer
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
                                        ('distance', pipe_distance, dist_cols)]
                                        ) # remainder='passthrough'

        # workflow
        pipe_cols = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                                    ('regressor', RandomForestRegressor())])

        return pipe_cols

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        
        scaled = pipeline.fit_transform(self.set_pipeline(X_train))
        model = pipeline.fit(scaled,y_train)
        return model

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.set_pipeline()
        trainer = self.run()
        y_pred = trainer.predict(pipeline.fit_transform(self.set_pipeline(X_test)))
        return mean_squared_error(y_test, y_pred, squared=False)
    
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("MLFLOW_URI")
        return MlflowClient()
    
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    EXPERIMENT_NAME ="86 SH LW model KAi + 1.01" 

    # Indicate mlflow to log to remote server
    MLFLOW_URI = "https://mlflow.lewagon.co/"

    client = MlflowClient()

    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    except BaseException:
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    yourname = 'KAi'

    if yourname is None:
        print("please define your name, il will be used as a parameter to log")

    for model in ["linear", "Randomforest"]:
        run = client.create_run(experiment_id)
        client.log_metric(run.info.run_id, "rmse", 4.5)
        client.log_param(run.info.run_id, "model", model)
        client.log_param(run.info.run_id, "student_name", yourname)
        
if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
