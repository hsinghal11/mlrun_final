import mlrun
from kfp import dsl

@dsl.pipeline(name="breast-cancer-pipeline")
def pipeline(model_name="breast-cancer-classifier"):

    # Data ingestion step
    ingest = mlrun.run_function(
        "load-breast-cancer-data",
        name="load-breast-cancer-data",
        params={"format": "pq", "model_name": model_name},
        outputs=["dataset"]
    )

    # Model training step with hyperparameter tuning
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        hyperparams={
            "n_estimators": [10, 100, 200],
            "max_depth": [2, 5, 10]
        },
        selector="max.accuracy",
        outputs=["model"]
    )

    # Model deployment step
    deploy = mlrun.deploy_function(
        "serving",
        models=[{
            "key": model_name,
            "model_path": train.outputs["model"],
            "class_name": "ClassifierModel"
        }],
        mock=True
    )