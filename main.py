import datetime

from cleaning.runner import Runner as CleaningPipelineRunner
from modelling.runner import Runner as ModelRunner

from heart_disease_analysis.preprocessing.runner import Runner as PreprocessingPipelineRunner
from heart_disease_analysis.utils import utils
from heart_disease_analysis.preprocessing import functions as ppf

config = utils.load_yaml("config.yaml")
run_ts = datetime.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
config["run_ts"] = run_ts

cleaning_pipeline = CleaningPipelineRunner(config["cleaning"])
cleaning_pipeline.run()

preprocessing_pipeline = PreprocessingPipelineRunner(config["preprocessing"])
preprocessing_pipeline.run()

variable_config = config["modelling"]["variable_config"]
features = (
    variable_config["continuous_variables"] + 
    list(variable_config["cols_to_label_encode"].keys()) + 
    ppf.get_one_hot_encoded_features_from_csv(
        config["preprocessing"]["output_path"], 
        variable_config["cols_to_one_hot_encode"]
    )
)
target = variable_config["target"]

modelling_pipeline = ModelRunner(config["modelling"], config["run_ts"], features, target)
results = modelling_pipeline.run()

utils.create_outputs(config, results)
