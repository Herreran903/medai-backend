#!/usr/bin/env python3
"""
Test script to verify LLM extractor integration with existing pipeline.
This script tests compatibility without requiring actual API calls.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.llm import LLMExtractor, ENTITY_CATEGORIES, LABEL_TO_CATEGORY
from app.schemas import Entity


def test_entity_categories():
    """Test that all entity categories are properly defined."""
    print("Testing entity categories...")
    
    assert len(ENTITY_CATEGORIES) == 6, "Should have 6 categories"
    
    categories = list(ENTITY_CATEGORIES.keys())
    expected = [
        "ventilacion",
        "respuesta_ventilacion", 
        "antropometricos",
        "signos_vitales",
        "observaciones",
        "gases_arteriales"
    ]
    
    for cat in expected:
        assert cat in categories, f"Missing category: {cat}"
    
    # Count total labels
    total_labels = sum(len(labels) for labels in ENTITY_CATEGORIES.values())
    print(f"✓ Total entity labels defined: {total_labels}")
    
    # Verify reverse mapping
    for category, labels in ENTITY_CATEGORIES.items():
        for label in labels:
            assert LABEL_TO_CATEGORY[label] == category, f"Mapping error for {label}"
    
    print("✓ Entity categories test passed\n")


def test_output_format_compatibility():
    """Test that LLM output format matches LSTM/Transformer format."""
    print("Testing output format compatibility...")
    
    # Mock entity output (what Claude would return after conversion)
    mock_entities = [
        {
            "type": "FiO2",
            "text": "60%",
            "start": 10,
            "end": 13,
            "score": 0.95,
            "code": "60"
        },
        {
            "type": "PEEP",
            "text": "10 cmH2O",
            "start": 20,
            "end": 28,
            "score": 0.92,
            "code": "10"
        }
    ]
    
    # Verify all required fields are present
    required_fields = ["type", "text", "start", "end", "score", "code"]
    for entity in mock_entities:
        for field in required_fields:
            assert field in entity, f"Missing required field: {field}"
    
    # Verify compatibility with Entity schema
    for entity_dict in mock_entities:
        try:
            entity = Entity(**entity_dict)
            assert entity.type == entity_dict["type"]
            assert entity.text == entity_dict["text"]
            assert entity.start == entity_dict["start"]
            assert entity.end == entity_dict["end"]
            assert entity.score == entity_dict["score"]
            print(f"✓ Entity validated: {entity.type} = {entity.text}")
        except Exception as e:
            print(f"✗ Entity validation failed: {e}")
            raise
    
    print("✓ Output format compatibility test passed\n")


def test_extractor_initialization():
    """Test that LLM extractor can be initialized without API key."""
    print("Testing extractor initialization...")
    
    # Test default initialization (Claude)
    extractor = LLMExtractor()
    assert extractor.provider == "claude"
    print(f"✓ Default provider: {extractor.provider}")
    
    # Test GPT stub
    extractor_gpt = LLMExtractor(provider="gpt")
    assert extractor_gpt.provider == "gpt"
    print(f"✓ GPT stub provider: {extractor_gpt.provider}")
    
    # Test local stub
    extractor_local = LLMExtractor(provider="local")
    assert extractor_local.provider == "local"
    print(f"✓ Local stub provider: {extractor_local.provider}")
    
    # Test meta method
    meta = extractor.meta()
    assert "facade_provider" in meta
    assert "extractor" in meta
    print(f"✓ Meta data: {meta}")
    
    print("✓ Extractor initialization test passed\n")


def test_predict_without_api_key():
    """Test that predict method handles missing API key gracefully."""
    print("Testing predict without API key...")
    
    extractor = LLMExtractor()
    text = "FiO2 60%, PEEP 10 cmH2O"
    
    # Should return empty list without crashing
    result = extractor.predict(text)
    assert isinstance(result, list), "Should return a list"
    print(f"✓ Predict returned: {result} (expected empty without API key)")
    
    print("✓ Predict without API key test passed\n")


def test_registry_compatibility():
    """Test that LLMExtractor can be used in MODEL_REGISTRY."""
    print("Testing registry compatibility...")
    
    from app.services.registry import MODEL_REGISTRY
    
    # Verify llm is in registry
    assert "llm" in MODEL_REGISTRY, "LLM should be in MODEL_REGISTRY"
    
    # Get the extractor
    llm_extractor = MODEL_REGISTRY["llm"]
    
    # Verify it has predict method
    assert hasattr(llm_extractor, "predict"), "Should have predict method"
    
    # Verify it has meta method
    assert hasattr(llm_extractor, "meta"), "Should have meta method"
    
    print(f"✓ LLM extractor in registry: {type(llm_extractor).__name__}")
    print("✓ Registry compatibility test passed\n")


def test_clinical_entity_labels():
    """Test specific clinical entity labels are present."""
    print("Testing clinical entity labels...")
    
    # Test ventilation entities
    ventilation_labels = ENTITY_CATEGORIES["ventilacion"]
    assert "FiO2" in ventilation_labels
    assert "PEEP" in ventilation_labels
    assert "volumen_tidal" in ventilation_labels
    print(f"✓ Ventilation labels: {len(ventilation_labels)} entities")
    
    # Test vital signs
    vital_signs = ENTITY_CATEGORIES["signos_vitales"]
    assert "temperatura" in vital_signs
    assert "frecuencia_cardiaca" in vital_signs
    print(f"✓ Vital signs labels: {len(vital_signs)} entities")
    
    # Test arterial gases
    gases = ENTITY_CATEGORIES["gases_arteriales"]
    assert "pH" in gases
    assert "PCO2" in gases
    assert "PaFi" in gases
    print(f"✓ Arterial gases labels: {len(gases)} entities")
    
    print("✓ Clinical entity labels test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Extractor Integration Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_entity_categories,
        test_output_format_compatibility,
        test_extractor_initialization,
        test_predict_without_api_key,
        test_registry_compatibility,
        test_clinical_entity_labels,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed! LLM extractor is compatible with the pipeline.")
        sys.exit(0)


if __name__ == "__main__":
    main()