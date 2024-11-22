import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
from sklearn.pipeline import Pipeline

# 1. Función de carga y limpieza de datos
def cargar_limpiar_datos(ruta):
    # Cargar datos
    df = pd.read_csv(ruta)
    
    # Verificar valores NaN
    print("Valores NaN por columna:")
    print(df.isnull().sum())
    
    # Estrategias de manejo de NaN
    # 1. Eliminar columnas con demasiados valores NaN (más del 50%)
    columnas_validas = df.columns[df.isnull().mean() < 0.5]
    df = df[columnas_validas]
    
    # 2. Para el resto, imputar valores
    # Imputación por media para numéricas
    # Imputación por moda para categóricas
    for columna in df.select_dtypes(include=['float64', 'int64']).columns:
        df[columna].fillna(df[columna].median(), inplace=True)
    
    for columna in df.select_dtypes(include=['object']).columns:
        df[columna].fillna(df[columna].mode()[0], inplace=True)
    
    return df

# 2. Análisis Inicial de Datos
def analisis_inicial(df):
    print("Información Básica del Dataset:")
    print(df.info())
    print("\nDistribución de Fraudes:")
    print(df['fraud'].value_counts(normalize=True))

# 3. Preprocesamiento
def preprocesar_datos(df):
    # Separar features y target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# 4. Definición de Modelos Base
def modelo_base(y_test):
    # Clasificador aleatorio como baseline
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(np.zeros((len(y_test), 1)), y_test)
    return dummy

# 5. Entrenamiento de Modelos
def entrenar_modelos(X_train, X_test, y_train, y_test):
    # Pipelines con escalamiento
    modelos = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ])
    }
    
    # Hiperparámetros para búsqueda
    parametros = {
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        # Grid Search con Cross-Validation
        grid = GridSearchCV(
            modelo, 
            parametros[nombre], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        # Mejores parámetros
        print(f"Mejores parámetros para {nombre}:")
        print(grid.best_params_)
        
        # Predicciones
        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]
        
        # Métricas de evaluación
        resultados[nombre] = {
            'clasificacion': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision_recall': average_precision_score(y_test, y_prob)
        }
    
    return resultados

# 6. Función Principal
def main():
    # Cargar y limpiar datos
    df = cargar_limpiar_datos('card_transdata.csv')
    
    # Análisis inicial
    analisis_inicial(df)
    
    # Preprocesamiento
    X_train, X_test, y_train, y_test = preprocesar_datos(df)
    
    # Modelo Base
    base = modelo_base(y_test)
    
    # Entrenar Modelos
    resultados = entrenar_modelos(X_train, X_test, y_train, y_test)
    
    # Imprimir Resultados
    for nombre, metricas in resultados.items():
        print(f"\nResultados para {nombre}:")
        print("Clasificación:")
        print(metricas['clasificacion'])
        print(f"ROC AUC: {metricas['roc_auc']}")
        print(f"Precision-Recall AUC: {metricas['precision_recall']}")

if __name__ == "__main__":
    main()
