import unittest

from src.gdaicommons.nlp.transformations import text_summarization


class TextSummarizationTest(unittest.TestCase):

    def test_text_summarization_single(self):
        config = {
            'model': "facebook/bart-large-cnn",
            'input_field': 'input_field',
            'output_field': 'text'
        }
        generator = text_summarization("my-text-generator", **config)

        test_data = [{'input_field': 'should i go to '}]
        generator(test_data)
        for data in test_data:
            self.assertTrue('text' in data)
            self.assertTrue(isinstance(data['text'], str))
            self.assertTrue(len(data['text']) > 0)

    def test_text_summarization_multiple(self):
        config = {
            'model': "facebook/bart-large-cnn",
            'input_field': 'input_field',
            'output_field': 'text'
        }
        generator = text_summarization("my-text-classifier", **config)

        test_data = [{'input_field': 'should i go to'}, {'input_field': 'i must go to'}]
        generator(test_data)
        for data in test_data:
            self.assertTrue('text' in data)
            self.assertTrue(isinstance(data['text'], str))
            self.assertTrue(len(data['text']) > 0)
