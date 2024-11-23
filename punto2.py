import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def remove_redundant_features(X, correlation_threshold=0.95):
    """Elimina características altamente correlacionadas."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    return X.drop(columns=to_drop), to_drop

def select_features(X, y, n_features=10):
    """Selecciona las características más importantes usando ANOVA F-value."""
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask].tolist()
    X_selected = pd.DataFrame(selector.transform(X), columns=selected_features)
    return X_selected, selected_features

class CreditCardDataset(Dataset):
    def __init__(self, X, y, is_multiclass=False):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X).to(device)
        self.y = (torch.LongTensor(y.values if isinstance(y, pd.Series) else y) if is_multiclass 
                 else torch.FloatTensor(y.values if isinstance(y, pd.Series) else y)).to(device)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelos
class LinearClassification(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassification, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

class LDAModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LDAModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        x = self.batch_norm(x)
        return self.linear(x)

def get_baseline_metrics(X_train, X_test, y_train, y_test):
    """Calcula métricas para un modelo baseline (clasificador aleatorio)."""
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict_proba(X_test)[:, 1]
    return {
        'roc_auc': roc_auc_score(y_test, y_pred),
        'pr_auc': average_precision_score(y_test, y_pred),
        'classification_report': classification_report(y_test, (y_pred > 0.5).astype(int))
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, model_name=""):
    best_val_loss = float('inf')
    best_metrics = None
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(outputs, batch_y)
                predictions = torch.softmax(outputs, dim=1)[:, 1]
            else:
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                predictions = outputs
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                if isinstance(criterion, nn.CrossEntropyLoss):
                    val_loss += criterion(outputs, batch_y).item()
                    predictions = torch.softmax(outputs, dim=1)[:, 1]
                else:
                    outputs = outputs.squeeze()
                    val_loss += criterion(outputs, batch_y).item()
                    predictions = outputs
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            predictions = np.array(all_preds)
            true_labels = np.array(all_labels)
            best_metrics = {
                'roc_auc': roc_auc_score(true_labels, predictions),
                'pr_auc': average_precision_score(true_labels, predictions),
                'classification_report': classification_report(
                    true_labels, 
                    (predictions > 0.5).astype(int)
                )
            }
            
        print(f'{model_name} Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
    return best_metrics

def main():
    # 1. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv('card_transdata.csv')
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # 2. Eliminar información redundante
    print("Eliminando features redundantes...")
    X_cleaned, dropped_features = remove_redundant_features(X)
    print(f"Features eliminadas por correlación: {dropped_features}")
    
    # 3. Selección de variables
    print("Seleccionando features importantes...")
    X_selected, selected_features = select_features(X_cleaned, y)
    print(f"Features seleccionadas: {selected_features}")
    
    # 4. Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns)
    
    # 5. Split inicial para test final
    print("Separando datos en train y test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Obtener métricas baseline
    print("Calculando métricas baseline...")
    baseline_metrics = get_baseline_metrics(X_train, X_test, y_train, y_test)
    print("\nMétricas Baseline:")
    print(f"ROC AUC Baseline: {baseline_metrics['roc_auc']:.4f}")
    print(f"PR AUC Baseline: {baseline_metrics['pr_auc']:.4f}")
    
    # 7. Crear datasets y dataloaders
    batch_size = 1024
    
    # 8. Configurar y entrenar modelos
    models = {
        'Linear Classification': {
            'model': LinearClassification(X_train.shape[1]),
            'criterion': nn.BCELoss(),
            'is_multiclass': False
        },
        'Logistic Regression': {
            'model': LogisticRegression(X_train.shape[1]),
            'criterion': nn.BCELoss(),
            'is_multiclass': False
        },
        'Multiclass Logistic': {
            'model': MulticlassLogisticRegression(X_train.shape[1], 2),
            'criterion': nn.CrossEntropyLoss(),
            'is_multiclass': True
        },
        'LDA': {
            'model': LDAModel(X_train.shape[1], 2),
            'criterion': nn.CrossEntropyLoss(),
            'is_multiclass': True
        }
    }
    
    # Entrenar y evaluar cada modelo
    results = {}
    
    for name, config in models.items():
        print(f"\nEntrenando {name}...")
        model = config['model'].to(device)
        criterion = config['criterion']
        is_multiclass = config['is_multiclass']
        
        # Crear datasets específicos para cada modelo
        train_dataset = CreditCardDataset(X_train, y_train, is_multiclass)
        test_dataset = CreditCardDataset(X_test, y_test, is_multiclass)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        results[name] = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            epochs=10,
            model_name=name
        )
        
        print(f"\nResultados para {name}:")
        print("Classification Report:")
        print(results[name]['classification_report'])
        print(f"ROC AUC: {results[name]['roc_auc']:.4f}")
        print(f"PR AUC: {results[name]['pr_auc']:.4f}")
    
    # 9. Comparar todos los modelos
    print("\nComparación final de modelos:")
    print("-----------------------------")
    print("\nMétricas Baseline:")
    print(f"ROC AUC Baseline: {baseline_metrics['roc_auc']:.4f}")
    print(f"PR AUC Baseline: {baseline_metrics['pr_auc']:.4f}")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()