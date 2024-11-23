import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset personalizado
class CreditCardDataset(Dataset):
    def __init__(self, X, y, is_multiclass=False):
        self.X = torch.FloatTensor(X).to(device)
        # Usar LongTensor para multiclass, FloatTensor para binario
        self.y = torch.LongTensor(y).to(device) if is_multiclass else torch.FloatTensor(y).to(device)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelo de Regresión Lineal para Clasificación
class LinearClassification(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassification, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Modelo de Regresión Logística
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Modelo de Regresión Logística Multiclase
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)  # No aplicamos softmax aquí ya que CrossEntropyLoss lo incluye

# Modelo de LDA
class LDAModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LDAModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        x = self.batch_norm(x)
        return self.linear(x)  # No aplicamos softmax aquí ya que CrossEntropyLoss lo incluye

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, model_name=""):
    best_val_loss = float('inf')
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Manejar diferentes tipos de salidas según el criterio
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
    # Cargar y preparar datos
    df = pd.read_csv('card_transdata.csv')
    X = df.drop('fraud', axis=1).values
    y = df['fraud'].values
    
    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crear datasets y dataloaders
    batch_size = 1024
    
    # Configurar modelos y sus respectivos datasets
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

if __name__ == "__main__":
    main()