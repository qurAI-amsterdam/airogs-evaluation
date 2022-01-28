import random
from tqdm import tqdm
import json
from copy import deepcopy
import math
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="preliminary_1")
args = parser.parse_args()
phase = args.phase

# Set random seed
random.seed(0)

single_example = {
    "inputs": [
        {
            "pk": 266510,
            "file": None,
            "image": {
                "pk": "46140e02-fc96-4ad1-ad81-a87ac16cb336",
                "name": "TRAIN000740.jpg"
            },
            "value": None,
            "interface": {
                "pk": 40,
                "kind": "Image",
                "slug": "color-fundus-image",
                "title": "Color Fundus Image",
                "super_kind": "Image",
                "description": "Retinal color fundus image (CFI)",
                "default_value": None,
                "relative_path": "images/color-fundus"
            }
        }
    ],
    "status": "Succeeded",
    "outputs": [
      {
        "value": [0.1829671208111137],
        "interface": {
            "slug": "multiple-referable-glaucoma-likelihoods"
        }
      },
      {
        "value": [False],
        "interface": {
            "slug": "multiple-referable-glaucoma-binary-decisions"
        }
      },
      {
        "value": [99],
        "interface": {
            "slug": "multiple-referable-ungradability-scores"
        }
      },
      {
        "value": [False],
        "interface": {
            "slug": "multiple-referable-ungradability-binary-decisions"
        }
      }
    ]
  }

group_size = 300

perfect = False

out = []
i = 0
reference = pd.read_csv(f'test/{phase}/reference.csv')
n = len(reference)
n_groups = math.ceil(n / group_size)
print(n)
# quit()
rng = np.random.RandomState(seed=42)

for group in range(n_groups):
  single_example_copy = deepcopy(single_example)
  single_example_copy["inputs"][0]["image"]["name"] = f'{group}.tiff'
  single_example_copy["outputs"][0]["value"] = []
  single_example_copy["outputs"][1]["value"] = []
  single_example_copy["outputs"][2]["value"] = []
  single_example_copy["outputs"][3]["value"] = []

  for j in tqdm(range(group_size)):
    ref = reference.iloc[i]['class']

    if perfect:
      rg_likelihood = int(ref == 'RG')
    else:
      rg_likelihood = abs(int(ref == 'RG') - rng.beta(2, 5))
    single_example_copy["outputs"][0]["value"].append(rg_likelihood)
    single_example_copy["outputs"][1]["value"].append(rg_likelihood >= .5)

    if perfect:
      u_score = int(ref == 'U')
      u_thresh = .5
    else:
      u_score = abs(int(ref == 'U') - rng.beta(2, 5)) * 6 - 1
      u_thresh = 2
    single_example_copy["outputs"][2]["value"].append(u_score)
    single_example_copy["outputs"][3]["value"].append(u_score >= u_thresh)

    i += 1

    if i >= len(reference):
      break
  out.append(single_example_copy)


# Dump json with indent 4
with open(f'test/{phase}/predictions.json', 'w') as f:
  f.write(json.dumps(out, indent=4))
