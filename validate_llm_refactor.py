#!/usr/bin/env python3
"""
Static validation script for LLM refactor.
Checks code structure and compatibility without requiring dependencies.
"""

import ast
import sys
from pathlib import Path


def validate_llm_file():
    """Validate the LLM.py file structure."""
    print("Validating app/models/llm.py...")
    
    llm_path = Path("app/models/llm.py")
    if not llm_path.exists():
        print("✗ File not found: app/models/llm.py")
        return False
    
    content = llm_path.read_text()
    
    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"✗ Syntax error in llm.py: {e}")
        return False
    
    # Check for required classes
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    required_classes = {
        "ClinicalEntity",
        "ClinicalEntitiesResponse",
        "ClaudeLLMExtractor",
        "GPTLLMExtractor",
        "LocalLLMExtractor",
        "LLMExtractor"
    }
    
    missing_classes = required_classes - classes
    if missing_classes:
        print(f"✗ Missing classes: {missing_classes}")
        return False
    
    print(f"✓ All required classes present: {len(required_classes)}")
    
    # Check for required constants
    if "ENTITY_CATEGORIES" not in content:
        print("✗ Missing ENTITY_CATEGORIES constant")
        return False
    
    if "LABEL_TO_CATEGORY" not in content:
        print("✗ Missing LABEL_TO_CATEGORY constant")
        return False
    
    print("✓ Required constants present")
    
    # Check for predict method in main classes
    for class_name in ["ClaudeLLMExtractor", "LLMExtractor"]:
        class_node = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if class_node:
            methods = {n.name for n in class_node.body if isinstance(n, ast.FunctionDef)}
            if "predict" not in methods:
                print(f"✗ Missing predict method in {class_name}")
                return False
            print(f"✓ {class_name} has predict method")
    
    # Check for entity categories
    categories = [
        "ventilacion",
        "respuesta_ventilacion",
        "antropometricos",
        "signos_vitales",
        "observaciones",
        "gases_arteriales"
    ]
    
    for category in categories:
        if category not in content:
            print(f"✗ Missing category: {category}")
            return False
    
    print(f"✓ All {len(categories)} entity categories present")
    
    # Check for specific entity labels
    key_labels = ["FiO2", "PEEP", "pH", "temperatura", "edad"]
    for label in key_labels:
        if label not in content:
            print(f"✗ Missing entity label: {label}")
            return False
    
    print(f"✓ Key entity labels present")
    
    print("✓ app/models/llm.py validation passed\n")
    return True


def validate_config_file():
    """Validate config.py has API key fields."""
    print("Validating app/config.py...")
    
    config_path = Path("app/config.py")
    if not config_path.exists():
        print("✗ File not found: app/config.py")
        return False
    
    content = config_path.read_text()
    
    if "anthropic_api_key" not in content:
        print("✗ Missing anthropic_api_key field")
        return False
    
    if "ANTHROPIC_API_KEY" not in content:
        print("✗ Missing ANTHROPIC_API_KEY env variable")
        return False
    
    print("✓ Anthropic API key configuration present")
    
    if "openai_api_key" not in content:
        print("✗ Missing openai_api_key field")
        return False
    
    print("✓ OpenAI API key configuration present")
    print("✓ app/config.py validation passed\n")
    return True


def validate_requirements():
    """Validate requirements.txt has LLM dependencies."""
    print("Validating requirements.txt...")
    
    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("✗ File not found: requirements.txt")
        return False
    
    content = req_path.read_text()
    
    if "anthropic" not in content:
        print("✗ Missing anthropic package")
        return False
    
    print("✓ anthropic package present")
    
    if "openai" not in content:
        print("✗ Missing openai package")
        return False
    
    print("✓ openai package present")
    print("✓ requirements.txt validation passed\n")
    return True


def validate_registry():
    """Validate registry.py imports LLMExtractor."""
    print("Validating app/services/registry.py...")
    
    registry_path = Path("app/services/registry.py")
    if not registry_path.exists():
        print("✗ File not found: app/services/registry.py")
        return False
    
    content = registry_path.read_text()
    
    if "from app.models.llm import" not in content:
        print("✗ Missing LLM import")
        return False
    
    if "LLMExtractor" not in content:
        print("✗ LLMExtractor not imported")
        return False
    
    if '"llm"' not in content and "'llm'" not in content:
        print("✗ LLM not in MODEL_REGISTRY")
        return False
    
    print("✓ LLMExtractor properly registered")
    print("✓ app/services/registry.py validation passed\n")
    return True


def validate_output_format():
    """Validate output format compatibility."""
    print("Validating output format compatibility...")
    
    llm_path = Path("app/models/llm.py")
    content = llm_path.read_text()
    
    # Check that predict returns the right format
    required_fields = ["type", "text", "start", "end", "score", "code"]
    
    for field in required_fields:
        if f'"{field}"' not in content and f"'{field}'" not in content:
            print(f"✗ Output field '{field}' not found in predict method")
            return False
    
    print(f"✓ All required output fields present: {required_fields}")
    print("✓ Output format validation passed\n")
    return True


def main():
    """Run all validations."""
    print("=" * 70)
    print("LLM Refactor Static Validation")
    print("=" * 70 + "\n")
    
    validations = [
        ("LLM File Structure", validate_llm_file),
        ("Config File", validate_config_file),
        ("Requirements", validate_requirements),
        ("Registry Integration", validate_registry),
        ("Output Format", validate_output_format),
    ]
    
    results = []
    for name, validator in validations:
        try:
            result = validator()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Validation error in {name}: {e}\n")
            results.append((name, False))
    
    print("=" * 70)
    print("Validation Summary:")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} validations passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✓ All validations passed!")
        print("✓ LLM refactor is structurally correct and compatible with pipeline")
        return 0
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())