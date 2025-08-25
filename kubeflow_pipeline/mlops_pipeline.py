import kfp
from kfp import dsl


#### Components of pipeline
def data_processing_op():
    return dsl.ContainerOp(
        name="Data Processing",
        image="beflous/my-mlops-app:latest",
        command=["python", "src/data_processing.py"]
    )

def model_training_op():
    return dsl.ContainerOp(
        name="Model Training",
        image="beflous/my-mlops-app:latest",
        command=["python", "src/model_training.py"]
    )

def launch_app_op():
    return dsl.ContainerOp(
        name="Launch App",
        image="beflous/my-mlops-app:latest",
        command=["python", "application.py"]
    )

### Pipeline starts here

@dsl.pipeline(
    name="MLOps Pipeline",
    description="This is my first ever kubeflow pipeline"
)

def mlops_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)
    launch_app = launch_app_op().after(model_training)

### RUN
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="mlops_pipeline.yaml"
    )
