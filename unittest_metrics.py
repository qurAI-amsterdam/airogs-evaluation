import unittest
from evaluation import screening_partial_auc, screening_sens_at_spec, ungradability_kappa, ungradability_auc, AIROGSReferenceLoader
import numpy as np
import tqdm
from sklearn import metrics


I_HAVE_PATIENCE = True


class TestScreeningPartialAUC(unittest.TestCase):
    def test_small_all_pred_neg(self):
        predictions_all = {'multiple-referable-glaucoma-likelihoods': [0, 0, 0, 0, 0]}

        reference_all = {
            '1': 'NRG',
            '2': 'RG',
            '3': 'RG',
            '4': 'RG',
            '5': 'U',
        }

        pauc = screening_partial_auc(predictions_all, reference_all, .95)

        self.assertEqual(pauc, 0.5)

    def test_large_all_pred_neg(self):
        reference_all = AIROGSReferenceLoader().load()

        predictions_all = {'multiple-referable-glaucoma-likelihoods': [0] * len(reference_all)}

        pauc = screening_partial_auc(predictions_all, reference_all, .95)
        self.assertEqual(pauc, 0.5)

        predictions_all = {
            k: {'multiple-referable-glaucoma-likelihoods': np.random.uniform(0, 1)} for k, v in reference_all.items()
        }

        predictions_all = {'multiple-referable-glaucoma-likelihoods': np.random.uniform(0, 1, len(reference_all))}
        pauc = screening_partial_auc(predictions_all, reference_all, .95)
        self.assertTrue(abs(pauc - 0.5) < .01)

    def test_all_pred_pos(self):

        predictions_all = {'multiple-referable-glaucoma-likelihoods': [1, 1, 1, 1]}

        reference_all = {
            '1': 'NRG',
            '2': 'RG',
            '3': 'RG',
            '4': 'RG',
        }

        pauc = screening_partial_auc(predictions_all, reference_all, .95)

        self.assertEqual(pauc, 0.5)

class TestScreeningSensAtSpec(unittest.TestCase):
    def test_random_example(self, n=10, seed_y_true=42, seed_y_pred=42):
        if seed_y_true == 'challenge_data':
            y_true = np.array([a == 'RG' for a in AIROGSReferenceLoader().load().values() if a != 'U'])
        else:
            y_true = np.random.RandomState(seed_y_true).choice([False, True], n)
        y_pred = np.random.RandomState(seed_y_pred).uniform(0, 1, len(y_true))

        reference_all = {
            str(i): 'RG' if yt else 'NRG' for i, yt in enumerate(y_true)
        }

        predictions_all = {'multiple-referable-glaucoma-likelihoods': list(y_pred)}

        threshes = list(sorted(y_pred))

        spec_to_best_sens = {}
        # current_highest_spec = -1
        for thresh in threshes:
            y_pred_t = y_pred >= thresh

            spec = np.sum((y_true == y_pred_t) & ~y_true) / np.sum(~y_true)
            sens = np.sum((y_true == y_pred_t) & y_true) / np.sum(y_true)

            if np.isnan(spec):
                continue

            if spec not in spec_to_best_sens:
                spec_to_best_sens[spec] = sens
            else:
                spec_to_best_sens[spec] = max(sens, spec_to_best_sens[spec])

        tqdm_maybe = lambda a: tqdm.tqdm(a) if len(a) > 1000 else a

        for spec, sens in tqdm_maybe(spec_to_best_sens.items()):
            sens_at_spec = screening_sens_at_spec(predictions_all, reference_all, spec)
            
            if np.isnan(sens_at_spec) and np.isnan(sens):
                self.assertTrue(True)
            else:
                self.assertAlmostEqual(sens_at_spec, sens, places=7)
    
    def test_random_examples(self):        
        for n in (
            (tqdm.tqdm([0, 1, 2, 3, 4, 5] + list(range(10, 100, 10)) + list(range(100, 1001, 100)) + [9999])) 
            if I_HAVE_PATIENCE else [0, 1, 5, 1000]
            ):
            n_y_true_seeds = 10 if n < 1001 else 1
            for seed in range(n_y_true_seeds):
                self.test_random_example(n=n, seed_y_true=seed, seed_y_pred=seed)


class TestUngradabilityKappa(unittest.TestCase):
    def calc_and_check(self, predictions_all, reference_all, override=False, override_value=None):
        y_true = [v for v in reference_all.values()]
        y_pred = predictions_all['multiple-referable-ungradability-binary-decisions']
        kappa0 = ungradability_kappa(predictions_all, reference_all)
        print(y_true, y_pred)
        kappa1 = metrics.cohen_kappa_score(np.array(y_true) == 'U', y_pred) if not override else override_value
        self.assertAlmostEqual(kappa0, kappa1, places=7)

    def test_small0(self):
        predictions_all = {'multiple-referable-ungradability-binary-decisions': [0, 0, 0, 0]}

        reference_all = {
            '1': 'NRG',
            '2': 'U',
            '3': 'RG',
            '4': 'U',
        }
        
        self.calc_and_check(predictions_all, reference_all)

    def test_small1(self):
        predictions_all = {'multiple-referable-ungradability-binary-decisions': [1, 1, 1, 1]}

        reference_all = {
            '1': 'NRG',
            '2': 'U',
            '3': 'RG',
            '4': 'U',
        }
        
        self.calc_and_check(predictions_all, reference_all)

    def test_small_allcorrect0(self):
        predictions_all = {'multiple-referable-ungradability-binary-decisions': [0, 1, 1, 1, 0]}

        reference_all = {
            '1': 'RG',
            '2': 'U',
            '3': 'U',
            '4': 'U',
            '5': 'NRG',
        }

        self.calc_and_check(predictions_all, reference_all, override=True, override_value=1)
        
class TestUngradabilityAUC(unittest.TestCase):
    # The same for AUC and ungradability_auc and multiple-referable-ungradability-scores instead of multiple-referable-ungradability-binary-decisions
    def calc_and_check(self, predictions_all, reference_all, override=False):
        y_true = [v for v in reference_all.values()]
        y_pred = predictions_all['multiple-referable-ungradability-scores']
        auc0 = ungradability_auc(predictions_all, reference_all)
        auc1 = metrics.roc_auc_score(np.array(y_true) == 'U', y_pred) if not override else override
        self.assertAlmostEqual(auc0, auc1, places=7)

    def test_small0(self):
        predictions_all = {'multiple-referable-ungradability-scores': [0, 0, 0, 0]}

        reference_all = {
            '1': 'NRG',
            '2': 'U',
            '3': 'RG',
            '4': 'U',
        }
        
        self.calc_and_check(predictions_all, reference_all)

    def test_small1(self):
        predictions_all = {'multiple-referable-ungradability-scores': [1, 1, 1, 1]}

        reference_all = {
            '1': 'NRG',
            '2': 'U',
            '3': 'RG',
            '4': 'U',
        }
        
        self.calc_and_check(predictions_all, reference_all)

    def test_small_allcorrect0(self):
        predictions_all = {'multiple-referable-ungradability-scores': [1, 0, 0, 0, 1]}

        reference_all = {
            '1': 'RG',
            '2': 'U',
            '3': 'U',
            '4': 'U',
            '5': 'NRG',
        }

        self.calc_and_check(predictions_all, reference_all)


    def test_small_allcorrect1(self):
        predictions_all = {'multiple-referable-ungradability-scores': [.9, .1, .1, .1, .9]}

        reference_all = {
            '1': 'RG',
            '2': 'U',
            '3': 'U',
            '4': 'U',
            '5': 'NRG',
        }

        self.calc_and_check(predictions_all, reference_all)


    def test_small_allcorrect2(self):
        predictions_all = {'multiple-referable-ungradability-scores': [-900, 10, 10, 20, -10000]}

        reference_all = {
            '1': 'RG',
            '2': 'U',
            '3': 'U',
            '4': 'U',
            '5': 'NRG',
        }

        self.calc_and_check(predictions_all, reference_all, override=1)

if __name__ == '__main__':
    unittest.main()
