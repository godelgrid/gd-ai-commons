import unittest

from src.gdaicommons.transformations import text_classification


class TextClassificationTest(unittest.TestCase):

    def test_text_classification(self):
        config = {
            'model': "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            'input_field': 'input_field',
            'label_output_field': 'label',
            'score_output_field': 'score',
        }
        classifier = text_classification("my-text-classifier", **config)

        test_data = [{'input_field': 'is this positive'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('label' in data)
            self.assertTrue('score' in data)

    def test_text_classification_multi_score(self):
        config = {
            'model': "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            'top_k_scores': '3',
            'input_field': 'input_field',
            'full_output_field': 'all_scores',
        }
        classifier = text_classification("my-text-classifier", **config)

        test_data = [{'input_field': 'is this positive'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertEqual(2, len(data['all_scores']))
