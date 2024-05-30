import unittest

from src.gdaicommons.nlp.transformations import text_classification


class TextClassificationTest(unittest.TestCase):

    def test_text_classification_single(self):
        config = {
            'model': "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            'input_field': 'input_field',
            'output_field': 'all_scores'
        }
        classifier = text_classification("my-text-classifier", **config)

        test_data = [{'input_field': 'is this positive'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            all_scores = data['all_scores']
            self.assertTrue(isinstance(all_scores, list))
            self.assertTrue(2, len(all_scores))
            for label in all_scores:
                self.assertTrue(2, len(label))
                self.assertTrue('score' in label)
                self.assertTrue(isinstance(label['score'], float))
                self.assertTrue('label' in label)
                self.assertTrue(isinstance(label['label'], str))

    def test_text_classification_multiple(self):
        config = {
            'model': "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            'input_field': 'input_field',
            'output_field': 'all_scores'
        }
        classifier = text_classification("my-text-classifier", **config)

        test_data = [{'input_field': 'is this positive'}, {'input_field': 'is this negative'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            all_scores = data['all_scores']
            self.assertTrue(isinstance(all_scores, list))
            self.assertTrue(2, len(all_scores))
            for label in all_scores:
                self.assertTrue(2, len(label))
                self.assertTrue('score' in label)
                self.assertTrue(isinstance(label['score'], float))
                self.assertTrue('label' in label)
                self.assertTrue(isinstance(label['label'], str))
