import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_name = "test-experiment"

try:
    experiment_id = client.create_experiment(experiment_name)
except MlflowException:
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id


experiment = client.get_experiment_by_name(experiment_name)
# print(experiment)

run_id = None
# run_id = "16e0d18c91ed4f9e8d915353e4a0052f"

if not run_id:
    study_run = client.create_run(experiment_id=experiment_id)
    study_run_id = study_run.info.run_id
else:
    study_run_id = run_id

print(study_run_id)

local_path = "/tmp/test_hist.png"
fig, ax = plt.subplots()
ax.hist(np.random.rand(2000))
fig.savefig(local_path)


with mlflow.start_run(run_id=study_run_id):
    mlflow.log_metric("f-beta", 2 * np.random.rand())
    mlflow.log_metric("revenue", 2 * np.random.rand())
    mlflow.log_metric("test_accuracy", np.random.rand())
    mlflow.log_artifact(local_path)
    mlflow.set_tag("compare", np.random.choice(["a", "b", "c"]))

run = client.get_run(study_run_id)
print("COMPLETED")
print(run)
