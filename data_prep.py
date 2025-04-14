import mlrun
from sklearn.datasets import load_breast_cancer
import pandas as pd

@mlrun.handler(outputs=["dataset", "label_column"])
def breast_cancer_loader(context, format="csv"):
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    context.logger.info(f"Saving breast cancer dataset to {context.artifact_path}")
    context.log_dataset("breast_cancer_dataset", df=df, format=format, index=False)

    return df, "target"

if __name__ == "__main__":
    with mlrun.get_or_create_ctx("breast_cancer_generator", upload_artifacts=True) as context:
        breast_cancer_loader(context, context.get_param("format", "csv"))