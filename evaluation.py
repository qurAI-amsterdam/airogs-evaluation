from numpy import e
from sklearn import metrics

from evalutils import ClassificationEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import (
    NumberOfCasesValidator, ExpectedColumnNamesValidator
)
import json
import pandas as pd
import numpy as np
import sys
import argparse


class AIROGSPredictionsLoader():
    def __init__(self): 
        self.expected_keys = ["multiple-referable-glaucoma-likelihoods",
            "multiple-referable-glaucoma-binary-decisions",
            "multiple-referable-ungradability-scores",
            "multiple-referable-ungradability-binary-decisions"]


    def validate(self, predictions):
        # Predictions validation
        for prediction in predictions.values():
            for key in self.expected_keys:
                # all_keys_str = ', '.join(list(prediction.keys()))
                # print(f"The only keys in the prediction are {all_keys_str}.")
                # print(key)
                if key not in prediction:
                    all_keys_str = ', '.join(list(prediction.keys()))
                    raise ValueError(f"Missing output: {key}. The only keys in the prediction are {all_keys_str}.")

            # ===== multiple-referable-glaucoma-binary-decisions =====
            lst = prediction["multiple-referable-glaucoma-binary-decisions"]
            if type(lst) is not list:
                raise ValueError(f"multiple-referable-glaucoma-binary-decisions is not a list, it has type {type(lst)}.")

            for val in lst:
                if type(val) is not bool:
                    raise ValueError(f"a value in multiple-referable-glaucoma-binary-decisions is not a boolean, it is {val}, which has type {type(val)}.")
            
            # ===== multiple-referable-glaucoma-likelihoods =====
            lst = prediction["multiple-referable-glaucoma-likelihoods"]

            if type(lst) is not list:
                raise ValueError(f"multiple-referable-glaucoma-likelihoods is not a list, it has type {type(lst)}.")

            for val in lst:
                if val < 0 or val > 1:
                    raise ValueError(f"a value in multiple-referable-glaucoma-likelihoods is not in [0, 1], it is {val}.")
            
            # ===== multiple-referable-ungradability-scores =====
            lst = prediction["multiple-referable-ungradability-scores"]

            if type(lst) is not list:
                raise ValueError(f"multiple-referable-ungradability-scores is not a list, it has type {type(lst)}.")

            for val in lst:
                if type(val) not in [float, int]:
                    raise ValueError(f"a value in multiple-referable-ungradability-scores is not a float or int, it is {val}, which has type {type(val)}.")
            
            # ===== multiple-referable-ungradability-binary-decisions =====
            lst = prediction["multiple-referable-ungradability-binary-decisions"]

            if type(lst) is not list:
                raise ValueError(f"multiple-referable-ungradability-binary-decisions is not a list, it has type {type(lst)}.")

            for val in lst:
                if type(val) is not bool:
                    raise ValueError(f"a value in multiple-referable-ungradability-binary-decisions is not a boolean, it is {val}, which has type {type(val)}.")


    def flatten(self, predictions):
        predictions_flattened = {}
        predictions_sorted = sorted([(k, v) for k, v in predictions.items()], key=lambda a: int(a[0]))  # Combine all stacked tiffs
        for key in self.expected_keys:
            predictions_flattened[key] = [vi for _, v in predictions_sorted for vi in v[key]]

        return predictions_flattened


    def load(self):
        with open(f"/input/predictions.json") as f:
            entries = json.load(f)

        """Convert entries json to a dictionary `predictions_flattened`, which is expected to be in the form of:
        {
            "multiple-referable-glaucoma-likelihoods": [0.5, ...],
            "multiple-referable-glaucoma-binary-decisions": [True, ...],
            "multiple-referable-ungradability-scores": [0.5, ...],
            "multiple-referable-ungradability-binary-decisions": [True, ...]
        }"""

        predictions = {}
        for entry in entries:
            name = entry["inputs"][0]["image"]["name"].split('.')[0]  # Get the filename and remove the extension
            predictions[name] = {}

            if entry["status"] != "Succeeded":
                raise ValueError("At least one of the images was not processed properly.")
            
            for output in entry["outputs"]:
                predictions[name][output["interface"]["slug"]] = output["value"]

        self.validate(predictions)
        predictions_flattened = self.flatten(predictions)

        return predictions_flattened


class AIROGSReferenceLoader():
    def load(self):
        fname = 'test/reference.csv'
        reference_table = pd.read_csv(fname)
        
        # Convert table to dict with key challenge_id and value class
        reference_dict = {}
        for _, row in reference_table.iterrows():
            # Remove extension from row['class']
            reference_dict[row['challenge_id']] = row['class']
        
        return reference_dict


def get_screening_cases(predictions_all, reference_all):
    reference = {nr: (idx, ref) for idx, (nr, ref) in enumerate(reference_all.items()) if ref != 'U'}

    y_true = np.array([ref == 'RG' for nr, (idx, ref) in reference.items()])
    y_pred = np.array([predictions_all['multiple-referable-glaucoma-likelihoods'][idx] for nr, (idx, ref) in reference.items()])

    return y_true, y_pred


def get_cases_for_ungradability_evaluation(predictions_all, reference_all, use_ungradability_score=True):
    if use_ungradability_score:
        lbl = 'multiple-referable-ungradability-scores' 
    else:
        lbl = 'multiple-referable-ungradability-binary-decisions'
    
    reference = {nr: (idx, ref) for idx, (nr, ref) in enumerate(reference_all.items())}

    y_true = np.array([ref for nr, (idx, ref) in reference.items()])
    y_pred = np.array([predictions_all[lbl][idx] for nr, (idx, ref) in reference.items()])

    return y_true, y_pred


def screening_partial_auc(predictions_all, reference_all, min_spec):    
    y_true, y_pred = get_screening_cases(predictions_all, reference_all)

    return metrics.roc_auc_score(y_true, y_pred, max_fpr=(1 - min_spec))


def screening_sens_at_spec(predictions_all, reference_all, at_spec, eps=sys.float_info.epsilon): 
    y_true, y_pred = get_screening_cases(predictions_all, reference_all)

    fpr, tpr, threshes = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
    spec = 1 - fpr

    operating_points_with_good_spec = spec >= (at_spec - eps)
    max_tpr = tpr[operating_points_with_good_spec][-1]

    operating_point = np.argwhere(operating_points_with_good_spec).squeeze()[-1]
    operating_tpr = tpr[operating_point]

    assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
    assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
        f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

    return max_tpr


def ungradability_kappa(predictions_all, reference_all):
    y_true, y_pred = get_cases_for_ungradability_evaluation(predictions_all, reference_all, use_ungradability_score=False)
    
    return metrics.cohen_kappa_score(y_true == 'U', y_pred)


def ungradability_auc(predictions_all, reference_all):
    y_true, y_pred = get_cases_for_ungradability_evaluation(predictions_all, reference_all, use_ungradability_score=True)

    return metrics.roc_auc_score(y_true == 'U', y_pred)


class Phase(ClassificationEvaluation):
    def __init__(self):
        # Load reference table and predictions, set output path
        self.loader = AIROGSPredictionsLoader()
        self.predictions = self.loader.load()  # Will raise ValueError if not all images were processed properly.
        self.reference = AIROGSReferenceLoader().load()
        self.output_path = '/output/metrics.json'
    
    def evaluate(self, only_validate=False):
        # Compute all metrics

        if only_validate:
            aggregates = {
                "success": True
            }
        else:
            aggregates = {
                "screening_partial_auc_90_spec": screening_partial_auc(self.predictions, self.reference, 0.9),  # alpha
                "screening_sens_at_95_spec": screening_sens_at_spec(self.predictions, self.reference, 0.95),  # beta
                "ungradability_kappa": ungradability_kappa(self.predictions, self.reference),  # gamma
                "ungradability_auc": ungradability_auc(self.predictions, self.reference),  # delta
            }

        metrics = {
            "success": True,
            "aggregates": aggregates
        }
        
        # Save metrics to file
        with open(self.output_path, 'w') as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # boolean argument called --only_validate
    parser.add_argument("--only_validate", action="store_true")
    args = parser.parse_args()

    only_validate = args.only_validate

    Phase().evaluate(only_validate=only_validate)
