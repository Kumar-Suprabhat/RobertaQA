import argparse
import os
import warnings
import json
import random
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from transformers import AdamW
from transformers.optimization import get_constant_schedule
import torch
import mlflow
import tempfile


# Initialize MLflow
os.environ["MLFLOW_TRACKING_URI"] = "/path_to_the_folder/mlruns"
mlflow.set_tracking_uri(f'file://{os.environ["MLFLOW_TRACKING_URI"]}/')

# Disable tokenizers warnings when constructing pipelines
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

tokenizer=None
model=None

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
# Define args as an attribute dictionary
arguments = AttrDict({
    'experiments_name': 'QA_Model_Finetuning_Experiments',
    # 'output_dir':'test_roberta' ,
    'train_paths':['train.json'],
    'eval_paths':['eval.json'] ,
    'test_paths':['test.json'],
    'epochs':3,
    'learning_rate':2e-5,
    'batch_size':16,
    'weight_decay':0.01,
    'model_name':"deepset/roberta-base-squad2",
    'seed':42,
    'train_on_all': False,
    'reload_model_type': 'best',
    'artifact_dir':''
})


def merge_json_datasets(files, output_dir, output_file_name):
    merged_data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged_data.extend(data)

    # Use the directory of the first file to construct the output file path
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    print(f"Merged data saved to {output_file_path}")
    return output_file_path


def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def compute_scores_for_dataset(predictions_list, truths_list):
    def compute_exact_match(predictions, truths):
        exact_matches = [int(pred == truth) for pred, truth in zip(predictions, truths)]
        return sum(exact_matches) / len(exact_matches) * 100

    def compute_f1_score(true_answers, predictions):
        f1_scores = []
        for true_answer, prediction in zip(true_answers, predictions):
            true_words = set(true_answer.split())
            predicted_words = set(prediction.split())
            shared_words = true_words.intersection(predicted_words)
            precision = len(shared_words) / len(predicted_words) if len(predicted_words) > 0 else 0
            recall = len(shared_words) / len(true_words) if len(true_words) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1_score)
        return sum(f1_scores) / len(f1_scores) * 100

    def relaxed_metric(predicted_answers, actual_answers):
        relaxed_scores = [int(pred.lower() in truth.lower()) for pred, truth in zip(predicted_answers, actual_answers)]
        return sum(relaxed_scores) / len(relaxed_scores) * 100

    def super_relaxed_metric(predicted_answers, actual_answers):
        super_relaxed_scores = [int(pred.lower() in truth.lower() or truth.lower() in pred.lower()) for pred, truth in zip(predicted_answers, actual_answers)]
        return sum(super_relaxed_scores) / len(super_relaxed_scores) * 100

    em_score = compute_exact_match(predictions_list, truths_list)
    f1_score = compute_f1_score(truths_list, predictions_list)
    relaxed_score = relaxed_metric(predictions_list, truths_list)
    super_relaxed_score = super_relaxed_metric(predictions_list, truths_list)
    return em_score, f1_score, relaxed_score, super_relaxed_score

def predict_and_compute_scores(model, tokenizer, context_list, question_list, original_answer_list, compute_scores_for_dataset):
    predicted_answer_list = []
    for context, question, original_answer in zip(context_list, question_list, original_answer_list):
        inputs = tokenizer(question, context, return_tensors="pt")
        # Specify device as 'mps' (multi-processing service) or any specific GPU
        device = torch.device('cuda:mps' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        final_answer = tokenizer.decode(predict_answer_tokens)
        final_answer = final_answer.strip()
        predicted_answer_list.append(final_answer)
    em_score, f1_score, relaxed_score, super_relaxed_score = compute_scores_for_dataset(predicted_answer_list, original_answer_list)
    return em_score, f1_score, relaxed_score, super_relaxed_score


def add_new_row(data, epoch, learning_rate, train_loss, eval_loss,
                eval_em_scores, eval_f1_scores, eval_relaxed_scores, eval_super_relaxed_scores,
                train_em_scores, train_f1_scores, train_relaxed_scores, train_super_relaxed_scores,
                test_em_scores, test_f1_scores, test_relaxed_scores, test_super_relaxed_scores,
                train_runtime, train_samples_per_second, train_steps_per_second,
                eval_runtime, eval_samples_per_second, eval_steps_per_second):
    new_row = {
        'epoch': epoch,
        'learning_rate': learning_rate,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'eval_em_scores': eval_em_scores,
        'eval_f1_scores': eval_f1_scores,
        'eval_relaxed_scores': eval_relaxed_scores,
        'eval_super_relaxed_scores': eval_super_relaxed_scores,
        'train_em_scores': train_em_scores,
        'train_f1_scores': train_f1_scores,
        'train_relaxed_scores': train_relaxed_scores,
        'train_super_relaxed_scores': train_super_relaxed_scores,
        'test_exact_match_score': test_em_scores,
        'test_f1_score': test_f1_scores,
        'test_relaxed_metric_score': test_relaxed_scores,
        'test_super_relaxed_metric_score': test_super_relaxed_scores,
        'train_runtime': train_runtime,
        'train_samples_per_second': train_samples_per_second,
        'train_steps_per_second': train_steps_per_second,
        'eval_runtime': eval_runtime,
        'eval_samples_per_second': eval_samples_per_second,
        'eval_steps_per_second': eval_steps_per_second
    }
    data.append(new_row)
    return data


def load_data(args):
    try:
        output_dir=args.artifact_dir
        # Convert Excel to SQuAD JSON
        train_paths= args.train_paths
        eval_paths = args.eval_paths
        test_paths = args.test_paths

        # Merge datasets
        train_final_path = merge_json_datasets(train_paths, output_dir, "merged_train_dataset.json")
        eval_final_path = merge_json_datasets(eval_paths, output_dir, "merged_eval_dataset.json")
        test_final_path = merge_json_datasets(test_paths, output_dir, "merged_test_dataset.json")

        # Load training data from JSON file
        with open(train_final_path, 'r', encoding='utf-8') as train_file:
            train_data = json.load(train_file)

        # Load evaluation data from JSON file
        with open(eval_final_path, 'r', encoding='utf-8') as eval_file:
            eval_data = json.load(eval_file)

        # Load test data from JSON file
        with open(test_final_path, 'r', encoding='utf-8') as test_file:
            test_data = json.load(test_file)
    except Exception as err:
        raise Exception(f"Failed to marge and load data | Error = {err}")
    return train_data, eval_data, test_data


def prepare_data(train_data, eval_data, test_data, tokenizer):
    try:
        # Define features dictionaries for train, eval, and test datasets
        train_features = {
            'question': [],
            'id': [],
            'context': [],
            'answers': [],
            'title': []
        }

        eval_features = {
            'question': [],
            'id': [],
            'context': [],
            'answers': [],
            'title': []
        }

        test_features = {
            'question': [],
            'id': [],
            'context': [],
            'answers': [],
            'title': []
        }

        # Populate features for train dataset
        for instance in train_data:
            for feature, value in instance.items():
                train_features[feature].append(value)

        # Populate features for eval dataset
        for instance in eval_data:
            for feature, value in instance.items():
                eval_features[feature].append(value)

        # Populate features for test dataset
        for instance in test_data:
            for feature, value in instance.items():
                test_features[feature].append(value)

        # Create Dataset objects for train, eval, and test datasets
        train_dataset = Dataset.from_dict(train_features)
        eval_dataset = Dataset.from_dict(eval_features)
        test_dataset = Dataset.from_dict(test_features)

        # Create DatasetDict objects for train and eval datasets
        dataset = DatasetDict({
            'train': train_dataset,
            'eval': eval_dataset,
            'test': test_dataset
        })
        '''
        # Create DatasetDict object for test dataset
        test_dataset = DatasetDict({
            'test': test_dataset
        })
        '''
        print(dataset)
        #print(test_dataset)

        # Preprocess datasets
        tokenized_squad = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

        print(dataset)
        print(test_dataset)
    except Exception as err:
        raise Exception(f"Failed to prepare data | Error = {err}")
    return dataset, tokenized_squad


def create_model_and_tokenizer(args):
    try:
        # Create model and tokenizer instances
        global model
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        global tokenizer
        tokenizer= AutoTokenizer.from_pretrained(args.model_name)

        # Set trainable attribute of model layers as per configuration
        model_type_full = args.model_name.split('/')[1]
        # Split the resulting part by '-' and take the first part
        model_type = model_type_full.split('-')[0]

        if not args.train_on_all:
            model_module = getattr(model, model_type)
            for param in model_module.parameters():
                param.requires_grad = False
    except Exception as err:
        raise Exception(f"Failed to create model and tokenizer instances for training | Error = {err}")
    return model, tokenizer


def load_model_and_tokenizer(args):
    try:
        
        filter_string = f'tags.mlflow.runName = "{get_runName(args)}"'
        df_runs = mlflow.search_runs(
            filter_string=filter_string,
            experiment_ids=[str(get_experiment_id(args))]
        )
        print("load_model_and_tokenizer: df_runs.run_id = {}, filter_string = {}".format(df_runs.run_id, filter_string))
        
        '''
        df_runs = mlflow.search_runs(
            experiment_ids=["0"]
           )
        '''
        #import pdb; pdb.set_trace()
        
        if args.reload_model_type.lower() == "best":
            df_runs.sort_values(['metrics.highest_f1_score'], ascending=False, inplace=True)
        elif args.reload_model_type.lower() == "latest":
            df_runs.sort_values(['start_time'], ascending=False, inplace=True)
        else:
            err_msg = f"Invalid value for argument 'args.reload_model_type' = '{args.reload_model_type}'. Supported types: ['best', 'latest']"
            print(err_msg)
            raise ValueError(err_msg)
            
        # best_run_id = df_runs.loc[df_runs.index[0], 'run_id']
        # experiment_id = df_runs.loc[df_runs.index[0], 'experiment_id']
        best_run_id = ""
        for index, row in df_runs.iterrows():
            if args.reload_model_type.lower() == "best":
                best_model_path = os.path.join(os.environ["MLFLOW_TRACKING_URI"], str(row["experiment_id"]), str(row["run_id"]), "artifacts", "best_f1_score_model")
            elif args.reload_model_type.lower() == "latest":
                best_model_path = os.path.join(os.environ["MLFLOW_TRACKING_URI"], str(row["experiment_id"]), str(row["run_id"]), "artifacts", "last_epoch_model")
            if os.path.exists(best_model_path):
                    best_run_id = row["run_id"]
                    break
        
        # Create model and tokenizer instances
        global model
        model = AutoModelForQuestionAnswering.from_pretrained(best_model_path)
        global tokenizer
        tokenizer= AutoTokenizer.from_pretrained(args.model_name)
    except Exception as err:
        raise Exception(f"Failed to load finetuned model and tokenizer instances for training/inference | Error = {err}")
    return model, tokenizer


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


def main(args):
    #load_model_and_tokenizer(args)
    # Create run at the beginning
    '''experiment_id = get_experiment_id(args)
    tags = {
        "mlflow.modelName": args.model_name,
        "mlflow.runName": get_runName(args)
    }

    with mlflow.start_run(
            experiment_id=str(experiment_id), 
            tags=tags
        ) as run:
        run_id = run.info.run_id
        '''

        # mlflow.log_artifact(train_final_path)
        # mlflow.log_artifact(eval_final_path)
        # mlflow.log_artifact(test_final_path)
        #artifact_dir = mlflow.get_artifact_uri()
        #artifact_dir=artifact_dir.replace("file://", "")
        #print(f"artifacts directory = {artifact_dir}")

    # Merge, Save and Load data
    train_data, eval_data, test_data = load_data(args)
        
    # Create base model and tokenizer instances
    model, tokenizer = create_model_and_tokenizer(args)


    # Prepare data in finetuning format
    dataset, tokenized_squad = prepare_data(train_data, eval_data, test_data, tokenizer)



    # Preparing dataset for evaluation
    data_collator = DefaultDataCollator()
    data=[]
    context_eval_list = dataset['eval']['context']
    question_eval_list = dataset['eval']['question']
    original_eval_answer = dataset['eval']['answers']
    original_eval_answer_list=[]
    for i in range(0,len(original_eval_answer)):
        original_eval_answer_list.append(original_eval_answer[i]['text'][0])
        
    context_train_list = dataset['train']['context']
    question_train_list = dataset['train']['question']
    original_train_answer = dataset['train']['answers']
    original_train_answer_list=[]
    for i in range(0,len(original_train_answer)):
        original_train_answer_list.append(original_train_answer[i]['text'][0])

    context_test_list = dataset['test']['context']
    question_test_list = dataset['test']['question']
    original_test_answer = dataset['test']['answers']
    original_test_answer_list=[]
    for i in range(0,len(original_test_answer)):
        original_test_answer_list.append(original_test_answer[i]['text'][0])

    eval_em_scores, eval_f1_scores, eval_relaxed_scores, eval_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_eval_list, question_eval_list, original_eval_answer_list, compute_scores_for_dataset)
    train_em_scores, train_f1_scores, train_relaxed_scores, train_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_train_list, question_train_list, original_train_answer_list, compute_scores_for_dataset)
    test_em_scores, test_f1_scores, test_relaxed_scores, test_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_test_list, question_test_list, original_test_answer_list, compute_scores_for_dataset)
    
    data=add_new_row(data, 0, 0, 0, 0, eval_em_scores, eval_f1_scores, eval_relaxed_scores, eval_super_relaxed_scores,
            train_em_scores, train_f1_scores, train_relaxed_scores, train_super_relaxed_scores,
            test_em_scores, test_f1_scores, test_relaxed_scores, test_super_relaxed_scores,0,0,0,0,0,0)
    # Specify device as 'mps' (multi-processing service) or any specific GPU
    # device = torch.device('cuda:mps' if torch.cuda.is_available() else 'cpu')

    # Create the optimizer with the desired learning rate
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # Create a constant learning rate scheduler
    scheduler = get_constant_schedule(optimizer)
    
    highest_em_score_info = (float('-inf'), None)  # (score, epoch)
    highest_f1_score_info = (float('-inf'), None)  # (score, epoch)
    highest_relaxed_match_info = (float('-inf'), None)  # (score, epoch)
    highest_super_relaxed_match_info = (float('-inf'), None)  # (score, epoch)
    
    training_args = TrainingArguments(
        report_to="mlflow",
        output_dir=args.artifact_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        #use_mps_device=True,
        num_train_epochs=1,
        weight_decay=args.weight_decay,
        seed=args.seed,
        push_to_hub=False,
    )

    for epoch in range(1,args.epochs+1):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_squad["train"],
            eval_dataset=tokenized_squad["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler), 
        )
        model.train()
        trainer.train()
        model.eval()
        learning_rate=None
        print(f"the current running epoch is {epoch}")
        for param_group in trainer.optimizer.param_groups:
            learning_rate = param_group['lr']
        train_metrics = trainer.evaluate(eval_dataset=trainer.train_dataset, metric_key_prefix="train")
        eval_metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset, metric_key_prefix="eval")
        train_loss = train_metrics['train_loss']
        train_runtime = train_metrics['train_runtime']
        train_samples_per_second = train_metrics['train_samples_per_second']
        train_steps_per_second = train_metrics['train_steps_per_second']
        eval_loss = eval_metrics['eval_loss']
        eval_runtime = eval_metrics['eval_runtime']
        eval_samples_per_second = eval_metrics['eval_samples_per_second']
        eval_steps_per_second = eval_metrics['eval_steps_per_second']
        # Get the list of folders inside the main folder
        eval_em_scores, eval_f1_scores, eval_relaxed_scores, eval_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_eval_list, question_eval_list, original_eval_answer_list, compute_scores_for_dataset)
        train_em_scores, train_f1_scores, train_relaxed_scores, train_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_train_list, question_train_list, original_train_answer_list, compute_scores_for_dataset)
        test_em_scores, test_f1_scores, test_relaxed_scores, test_super_relaxed_scores=predict_and_compute_scores(model, tokenizer, context_test_list, question_test_list, original_test_answer_list, compute_scores_for_dataset)
        data=add_new_row(data, epoch, learning_rate, train_loss, eval_loss, eval_em_scores, eval_f1_scores, eval_relaxed_scores, eval_super_relaxed_scores,
        train_em_scores, train_f1_scores, train_relaxed_scores, train_super_relaxed_scores,
        test_em_scores, test_f1_scores, test_relaxed_scores, test_super_relaxed_scores, train_runtime, train_samples_per_second, train_steps_per_second, eval_runtime, eval_samples_per_second, eval_steps_per_second)
        if(eval_f1_scores > highest_f1_score_info[0]):
            model_path = os.path.join(args.artifact_dir, 'best_f1_score_model')
            model.save_pretrained(model_path)
            highest_f1_score_info = (eval_f1_scores, epoch)
            print('model has been saved for f-1')
            
        # Update highest relaxed match score and store the epoch number
        if eval_em_scores > highest_em_score_info[0]:
            highest_em_score_info = (eval_em_scores, epoch)
            
        # Update highest relaxed match score and store the epoch number
        if eval_relaxed_scores > highest_relaxed_match_info[0]:
            highest_relaxed_match_info = (eval_relaxed_scores, epoch)
            
        # Update highest super relaxed match score and store the epoch number
        if eval_super_relaxed_scores > highest_super_relaxed_match_info[0]:
            highest_super_relaxed_match_info = (eval_super_relaxed_scores, epoch)

        model_path = os.path.join(args.artifact_dir, 'last_epoch_model')
        model.save_pretrained(model_path)
        print('model has been saved for last epoch')
        
    # Log highest F1 score and its epoch
    mlflow.log_metric("highest_f1_score", highest_f1_score_info[0], step=highest_f1_score_info[1])

    # Log highest EM score (assuming you want the highest score tracked, update as needed)
    mlflow.log_metric("highest_em_score", highest_em_score_info[0], step=highest_em_score_info[1])  # Adjust as necessary

    # Log highest relaxed match score and its epoch
    mlflow.log_metric("highest_relaxed_match_score", highest_relaxed_match_info[0], step=highest_relaxed_match_info[1])

    # Log highest super relaxed match score and its epoch
    mlflow.log_metric("highest_super_relaxed_match_score", highest_super_relaxed_match_info[0], step=highest_super_relaxed_match_info[1])

    #print(f'the mlflow run id for this run is {run_id}')
    df = pd.DataFrame(data)

    # Save DataFrame to Excel
    excel_path = 'metrics_testing_final.xlsx'
    df.to_excel(excel_path, index=False)
    excel_path = os.path.abspath(excel_path)
    mlflow.log_artifact(excel_path)
    #csv_path = 'metrics_latest.csv'
    #df.to_csv(csv_path, index=False)
    print(f"DataFrame saved to {excel_path}")


if __name__ == "__main__":
    main(arguments)
    
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a QA model with SQuAD data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model and logs")
    parser.add_argument("--train_paths", nargs='+', required=True, help="List of train JSON file paths")
    parser.add_argument("--eval_paths", nargs='+', required=True, help="List of eval JSON file paths")
    parser.add_argument("--test_paths", nargs='+', required=True, help="List of test JSON file paths")
    parser.add_argument("--epochs", type=int,default=2, help="pass on the number of epochs")
    parser.add_argument("--learning_rate", type=float,default=2e-5, help="pass on the learning rate")
    parser.add_argument("--batch_size", type=int,default=16, help="pass on the batch size")
    parser.add_argument("--weight_decay", type=float,default=0.01, help="pass on the weight decay")
    parser.add_argument("--model_name", type=str, default="deepset/roberta-base-squad2", help="Model that shouls be loaded")
    parser.add_argument("--tokenizer_name", type=str, default="deepset/roberta-base-squad2", help="tokenizer that should be loaded")
    parser.add_argument("--seed", type=int, default=42, help="seed value to generate same values")
    parser.add_argument("--train_on_all", type=bool,default=False, help="pass on if you want to freeze layer")
    args = parser.parse_args()

    main(args)
'''