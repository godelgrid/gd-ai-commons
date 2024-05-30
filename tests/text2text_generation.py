import unittest

from src.gdaicommons.nlp.transformations import text2text_generation


class Text2TextGenerationTest(unittest.TestCase):

    def test_text_generation_single(self):
        config = {
            'model': "facebook/bart-large-cnn",
            'input_field': 'input_field',
            'output_field': 'text'
        }
        generator = text2text_generation("my-text-generator", **config)

        test_data = [{'input_field': 'should i go to '}]
        generator(test_data)
        for data in test_data:
            self.assertTrue('text' in data)
            self.assertTrue(isinstance(data['text'], str))
            self.assertTrue(len(data['text']) > 0)

    def test_text_generation_multiple(self):
        config = {
            'model': "mrm8488/t5-base-finetuned-question-generation-ap",
            'input_field': 'input_field',
            'output_field': 'text'
        }
        generator = text2text_generation("my-text-generator", **config)

        test_data = [{'input_field': 'should i go to'}, {'input_field': 'i must go to'}]
        generator(test_data)
        for data in test_data:
            self.assertTrue('text' in data)
            self.assertTrue(isinstance(data['text'], str))
            self.assertTrue(len(data['text']) > 0)
