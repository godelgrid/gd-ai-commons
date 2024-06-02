import unittest

from src.gdaicommons.constants import VALUE_SEPARATOR
from src.gdaicommons.nlp.transformations import text_labelling


class TextLabellingTest(unittest.TestCase):

    def test_text_labelling_single(self):
        config = {
            'model': "facebook/bart-large-mnli",
            'input_field': 'input_field',
            'labels': VALUE_SEPARATOR.join(['label1', 'label2', 'label3', 'label4']),
            'output_field': 'all_scores',
        }
        classifier = text_labelling("my-text-classifier", **config)

        test_data = [{'input_field': 'I have a problem with my iphone that needs to be resolved asap!!'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertTrue(isinstance(data['all_scores'], dict))
            self.assertTrue('labels' in data['all_scores'])
            self.assertTrue('scores' in data['all_scores'])
            self.assertEqual(2, len(data['all_scores']))
            self.assertEqual(4, len(data['all_scores']['labels']))
            self.assertEqual(4, len(data['all_scores']['scores']))

    def test_text_labelling_multiple(self):
        config = {
            'model': "facebook/bart-large-mnli",
            'input_field': 'input_field',
            'labels': VALUE_SEPARATOR.join(['label1', 'label2', 'label3', 'label4']),
            'output_field': 'all_scores',
        }
        classifier = text_labelling("my-text-classifier", **config)

        test_data = [{'input_field': 'I have a problem with my iphone that needs to be resolved asap!!'},
                     {'input_field': 'I have a problem with my iphone that needs to be resolved asap!!'}]
        classifier(test_data)
        for data in test_data:
            self.assertTrue('all_scores' in data)
            self.assertTrue(isinstance(data['all_scores'], dict))
            self.assertTrue('labels' in data['all_scores'])
            self.assertTrue('scores' in data['all_scores'])
            self.assertEqual(2, len(data['all_scores']))
            self.assertEqual(4, len(data['all_scores']['labels']))
            self.assertEqual(4, len(data['all_scores']['scores']))
