from unittest import TestCase
from src.base_model import BaseModel


class TestBaseModel(TestCase):
    def setUp(self):
        self.basemodel = BaseModel(10)
    def test_step(self):
        self.fail()


class TestBaseModel:
    def test_get_firm_size_distribution(self):
        assert False

    def test_step(self):
        assert False
