import pytest
import json
import tempfile
import os
import unittest
from unittest.mock import patch

from quantbullet.linear_product_model.component_manager import (
    ComponentConfig,
    ComponentRegistry,
    ComponentManager,
    dataclass_to_dict,
    dict_to_dataclass,
)
from quantbullet.parametric_model import AsymQuadModel, DoubleLogisticModel


class TestComponentConfig(unittest.TestCase):
    """Test ComponentConfig dataclass functionality."""
    def test_component_config_build(self):
        """Test building a model from ComponentConfig."""
        config = ComponentConfig(
            class_name="AsymQuadModel",
            args={"params_dict": {"a": 1.0, "b": 2.0, "x0": 0.0, "c": 1.0}}
        )
        
        model = config.build()
        self.assertIsInstance(model, AsymQuadModel)
        self.assertEqual(model.params_dict, {"a": 1.0, "b": 2.0, "x0": 0.0, "c": 1.0})


class TestComponentRegistry(unittest.TestCase):
    """Test ComponentRegistry dataclass functionality."""
    
    def test_component_registry_creation(self):
        """Test basic ComponentRegistry creation."""
        config1 = ComponentConfig("AsymQuadModel", {"param": "value1"})
        config2 = ComponentConfig("DoubleLogisticModel", {"param": "value2"})
        
        registry = ComponentRegistry(
            label="TestRegistry",
            components={"model1": config1, "model2": config2}
        )
        
        self.assertEqual(registry.label, "TestRegistry")
        self.assertEqual(len(registry.components), 2)
        self.assertIn("model1", registry.components)
        self.assertIn("model2", registry.components)

class TestComponentManager(unittest.TestCase):
    """Test ComponentManager class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ComponentManager()
        config1 = ComponentConfig(
            class_name="AsymQuadModel",
            args={"params_dict": {"a": 1.0, "b": 2.0, "x0": 0.0, "c": 1.0}}
        )
        config2 = ComponentConfig(
            class_name="DoubleLogisticModel",
            args={"params_dict": {"L1": 1.0, "L2": 2.0, "k1": 1.0, "x1": 0.0, 
                                  "k2": 1.0, "x2": 1.0, "c": 0.0}}
        )

        self.config1, self.config2 = config1, config2
    
    def test_build_all_models(self):
        """Test building all models from components."""
        # Add multiple components
        config1, config2 = self.config1, self.config2
        self.manager.add_component("asym_quad", config1)
        self.manager.add_component("double_logistic", config2)
        
        # Build all models
        self.manager.build_all_models()
        
        self.assertTrue(hasattr(self.manager, 'models'))
        self.assertEqual(len(self.manager.models), 2)
        self.assertIsInstance(self.manager.models["asym_quad"], AsymQuadModel)
        self.assertIsInstance(self.manager.models["double_logistic"], DoubleLogisticModel)
    
    def test_dict_roundtrip_serialization(self):
        """Test complete roundtrip: create -> serialize -> deserialize -> verify."""
        config1, config2 = self.config1, self.config2
        
        original = ComponentManager()
        original.registry.label = "RoundtripTest"
        original.add_component("quad", config1)
        original.add_component("logistic", config2)
        
        # Convert to dict and back
        data = original.to_dict()
        restored = ComponentManager.from_dict(data)
        
        # Verify everything matches
        self.assertEqual(restored.registry.label, original.registry.label)
        self.assertEqual(set(restored.component_names), set(original.component_names))
        
        for name in original.component_names:
            orig_config = original.get_component(name)
            rest_config = restored.get_component(name)
            
            self.assertEqual(orig_config.class_name, rest_config.class_name)
            self.assertEqual(orig_config.args, rest_config.args)
            self.assertEqual(orig_config.metadata, rest_config.metadata)

    def test_json_roundtrip_serialization(self):
        """Test complete roundtrip: create -> serialize -> deserialize -> verify, including file I/O."""
        config1, config2 = self.config1, self.config2
        
        original = ComponentManager()
        original.registry.label = "RoundtripTest"
        original.add_component("quad", config1)
        original.add_component("logistic", config2)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            original.to_json(tmp_path)
            file_restored = ComponentManager.from_json(tmp_path)

            self.assertEqual(file_restored.registry.label, original.registry.label)
            self.assertEqual(set(file_restored.component_names), set(original.component_names))

            for name in original.component_names:
                self.assertEqual(file_restored.get_component(name), original.get_component(name))
        finally:
            os.remove(tmp_path)