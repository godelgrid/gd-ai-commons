import unittest

from src.gdaicommons.nlp.transformations import fill_mask


class FillMaskTest(unittest.TestCase):

    def test_fill_mask_single(self):
        config = {
            'model': "google-bert/bert-base-uncased",
            'input_field': 'input_field',
            'output_field': 'all_scores',
        }
        classifier = fill_mask("my-text-classifier", **config)

        test_data = [{'input_field': 'This is a simple [MASK].'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertEqual(5, len(data['all_scores']))
            for label in data['all_scores']:
                self.assertTrue(isinstance(label, dict))
                self.assertEqual(3, len(label))
                self.assertTrue('token_str' in label)
                self.assertTrue(isinstance(label['token_str'], str))
                self.assertTrue('token' in label)
                self.assertTrue(isinstance(label['token'], int))
                self.assertTrue('score' in label)
                self.assertTrue(isinstance(label['score'], float))

    def test_fill_mask_multiple(self):
        config = {
            'model': "google-bert/bert-base-uncased",
            'input_field': 'input_field',
            'output_field': 'all_scores',
        }
        classifier = fill_mask("my-text-classifier", **config)

        test_data = [{'input_field': 'This is a simple [MASK].'}, {'input_field': 'This is a simple [MASK].'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertEqual(5, len(data['all_scores']))
            for label in data['all_scores']:
                self.assertTrue(isinstance(label, dict))
                self.assertEqual(3, len(label))
                self.assertTrue('token_str' in label)
                self.assertTrue(isinstance(label['token_str'], str))
                self.assertTrue('token' in label)
                self.assertTrue(isinstance(label['token'], int))
                self.assertTrue('score' in label)
                self.assertTrue(isinstance(label['score'], float))

    def test_fill_mask_custom(self):
        config = {
            'model': "google-bert/bert-base-uncased",
            'input_field': 'input_field',
            'mask_placeholder': '[MY_MASK]',
            'output_field': 'all_scores',
        }
        classifier = fill_mask("my-text-classifier", **config)

        test_data = [{'input_field': 'This is a simple [MY_MASK].'}, {'input_field': 'This is a simple [MY_MASK].'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertEqual(5, len(data['all_scores']))

    def test_fill_mask_custom_target(self):
        config = {
            'model': "google-bert/bert-base-uncased",
            'input_field': 'input_field',
            'mask_placeholder': '[MY_MASK]',
            'mask_targets': 'problem, question',
            'output_field': 'all_scores',
        }
        classifier = fill_mask("my-text-classifier", **config)

        test_data = [{'input_field': 'This is a simple [MY_MASK].'}, {'input_field': 'This is a simple [MY_MASK].'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertEqual(2, len(data['all_scores']))
            for label in data['all_scores']:
                self.assertTrue(label['token_str'] in ['problem', 'question'])
