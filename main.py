import sys
from pathlib import Path
import mlflow

# Add the directories containing the scripts to the system path if required
sys.path.append(str(Path("/home/ubuntu/kumar-suprabhat")))

# Import the specific functions needed
from data import main as data_preparation
from Roberta_QA import main as model_finetuning

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
# Combine the two dictionaries
args = AttrDict({
    'data_preparation': AttrDict({
        'prepare': AttrDict({'enabled': True, 'input': ['/predicted_results _qa_original.xlsx'], 'look_up_file': '/usr/context_data.json', 'create_context_using_chatgpt':False, 'row_name_parametername':'parameter_name', 'row_name_parametervalue':'parameter_value', 'row_name_context':'Text'}),
        'update': AttrDict({'enabled': False, 'input': ''}),
        'augment': AttrDict({'enabled': False, 'input': '', 'look_up_file': ''}),
        'split': AttrDict({'enabled': True, 'input': '', 'augment_file': ''}),
    }),
    'model_finetuning': AttrDict({
        'experiments_name': 'QA_Model_Finetuning_Experiments',
        'train_paths': [],
        'eval_paths': [],
        'test_paths': [],
        'epochs': 3,
        'learning_rate': 2e-5,
        'batch_size': 16,     
        'weight_decay': 0.01,
        'model_name': "deepset/roberta-base-squad2",
        'seed': 42,
        'train_on_all': False,
        'reload_model_type': 'best',
        'artifact_dir':'',
    })
})

def get_experiment_id(args):
    experiment = mlflow.get_experiment_by_name(args.experiments_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.experiments_name)
        experiment = mlflow.get_experiment(experiment_id)
        print(f"{args.experiments_name} experiment created with name = {experiment.name} and id = {experiment.experiment_id}")
    else:
        print(f"{args.experiments_name} experiment already exists name = {experiment.name} and id = {experiment.experiment_id}")
    return experiment.experiment_id


def get_runName(args):
    return f"{args.model_name}_finetuned"

# Now combined_args contains both data preparation and model finetuning configurations
def main():
      #make mlrun
    experiment_id = get_experiment_id(args.model_finetuning)
    tags = {
        "mlflow.modelName": args.model_finetuning.model_name,
        "mlflow.runName": get_runName(args.model_finetuning)
    }

    with mlflow.start_run(
            experiment_id=str(experiment_id), 
            tags=tags
        ) as run:
        run_id = run.info.run_id
        artifact_dir = mlflow.get_artifact_uri()
        artifact_dir=artifact_dir.replace("file://", "")
        print(f"artifacts directory = {artifact_dir}")
        args.model_finetuning.artifact_dir=artifact_dir
        #print(args.data_preparation)
        train_paths, eval_paths, test_paths=data_preparation(args.data_preparation)
        args.model_finetuning.train_paths.extend(train_paths)
        args.model_finetuning.eval_paths.extend(eval_paths)
        args.model_finetuning.test_paths.extend(test_paths)
        model_finetuning(args.model_finetuning)
        print(f'the run_id for this epoch is {run_id}')

if __name__ == "__main__":
    main()
