"""
Script de prueba para verificar la implementación de RoBERTa Transformer Extractor.
Prueba tanto la clase específica RobertaTransformerExtractor como el facade TransformerExtractor.
"""

import os
import sys

# Asegura que el directorio app esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.transformer import (
    RobertaTransformerExtractor,
    BETOTransformerExtractor,
    TransformerExtractor,
)


def test_roberta_direct():
    """Prueba directa de RobertaTransformerExtractor"""
    print("\n" + "="*80)
    print("TEST 1: RobertaTransformerExtractor (Directo)")
    print("="*80)
    
    try:
        # Inicializa el extractor RoBERTa
        extractor = RobertaTransformerExtractor()
        
        # Muestra metadatos
        meta = extractor.meta()
        print(f"\n✓ Extractor inicializado correctamente")
        print(f"  - Variante: {meta.get('variant')}")
        print(f"  - Extractor: {meta.get('extractor')}")
        print(f"  - Modelo: {meta.get('model_id')}")
        print(f"  - Base: {meta.get('base_model_id')}")
        print(f"  - Tokenización: {meta.get('tokenization')}")
        print(f"  - Arquitectura: {meta.get('architecture')}")
        print(f"  - Dispositivo: {meta.get('device')}")
        print(f"  - Etiquetas: {meta.get('num_labels')}")
        
        # Texto de prueba
        text = "Paciente con FiO2 al 60%, PEEP de 8 cmH2O, temperatura de 37.5°C"
        
        print(f"\n✓ Probando predicción con texto de ejemplo...")
        print(f"  Texto: '{text}'")
        
        # Realiza predicción
        entities = extractor.predict(text)
        
        print(f"\n✓ Predicción completada")
        print(f"  - Entidades encontradas: {len(entities)}")
        
        if entities:
            print("\n  Entidades extraídas:")
            for i, ent in enumerate(entities, 1):
                print(f"    {i}. {ent['type']}: '{ent['text']}' (pos: {ent['start']}-{ent['end']})")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error en prueba directa de RoBERTa: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_beto_direct():
    """Prueba directa de BETOTransformerExtractor para comparación"""
    print("\n" + "="*80)
    print("TEST 2: BETOTransformerExtractor (Comparación)")
    print("="*80)
    
    try:
        # Inicializa el extractor BETO
        extractor = BETOTransformerExtractor()
        
        # Muestra metadatos
        meta = extractor.meta()
        print(f"\n✓ Extractor BETO inicializado correctamente")
        print(f"  - Variante: {meta.get('variant')}")
        print(f"  - Extractor: {meta.get('extractor')}")
        print(f"  - Modelo: {meta.get('model_id')}")
        print(f"  - Base: {meta.get('base_model_id')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error en prueba de BETO: {e}")
        return False


def test_facade_roberta():
    """Prueba del facade TransformerExtractor con RoBERTa"""
    print("\n" + "="*80)
    print("TEST 3: TransformerExtractor Facade (RoBERTa)")
    print("="*80)
    
    try:
        # Inicializa con un model_id que contenga "roberta"
        extractor = TransformerExtractor(
            model_id="PlanTL-GOB-ES/roberta-base-bne"
        )
        
        # Verifica que detectó RoBERTa
        meta = extractor.meta()
        print(f"\n✓ Facade inicializado correctamente")
        print(f"  - Variante detectada: {meta.get('facade_variant')}")
        print(f"  - Variante interna: {meta.get('variant')}")
        print(f"  - Extractor: {meta.get('extractor')}")
        
        if meta.get('facade_variant') == 'roberta':
            print(f"\n✓ Facade detectó correctamente la variante RoBERTa")
            return True
        else:
            print(f"\n✗ Facade no detectó RoBERTa correctamente")
            return False
        
    except Exception as e:
        print(f"\n✗ Error en prueba de facade: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_facade_beto():
    """Prueba del facade TransformerExtractor con BETO"""
    print("\n" + "="*80)
    print("TEST 4: TransformerExtractor Facade (BETO)")
    print("="*80)
    
    try:
        # Inicializa con un model_id que contenga "beto"
        extractor = TransformerExtractor(
            model_id="NicolasUnivalle/beto-vm-ner-full"
        )
        
        # Verifica que detectó BETO
        meta = extractor.meta()
        print(f"\n✓ Facade inicializado correctamente")
        print(f"  - Variante detectada: {meta.get('facade_variant')}")
        print(f"  - Variante interna: {meta.get('variant')}")
        print(f"  - Extractor: {meta.get('extractor')}")
        
        if meta.get('facade_variant') == 'beto':
            print(f"\n✓ Facade detectó correctamente la variante BETO")
            return True
        else:
            print(f"\n✗ Facade no detectó BETO correctamente")
            return False
        
    except Exception as e:
        print(f"\n✗ Error en prueba de facade BETO: {e}")
        return False


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*80)
    print("PRUEBAS DE IMPLEMENTACIÓN DE RoBERTa")
    print("="*80)
    
    results = {
        "RoBERTa Directo": test_roberta_direct(),
        "BETO Directo": test_beto_direct(),
        "Facade RoBERTa": test_facade_roberta(),
        "Facade BETO": test_facade_beto(),
    }
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE PRUEBAS")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASÓ" if passed else "✗ FALLÓ"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n✓ ¡Todas las pruebas pasaron exitosamente!")
        return 0
    else:
        print(f"\n✗ {total - passed} prueba(s) fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())