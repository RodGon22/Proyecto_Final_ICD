"""
20 de noviembre del 2025
Rodrigo Gonzaga Sierra
Procesamiento de imagenes
PCA, NMF y clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Configuracion de estilo
plt.style.use('default')
sns.set_palette("husl")

class ImageProcessor:
    """Clase para procesamiento y analisis de imagenes"""
    
    def __init__(self):
        self.datasets = {}
        self.results = {}
    
        #Cargar dataset MNIST
    def load_mnist(self):
        print("Cargando MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.datasets['mnist'] = {
            'data': mnist.data,
            'target': mnist.target.astype(int),
            'target_names': np.array([str(i) for i in range(10)]),
            'shape': (28, 28),
            'name': 'MNIST Digits'
        }
        print(f"MNIST cargado: {mnist.data.shape[0]} imágenes de {mnist.data.shape[1]} pixels")
        
        #Cargar dataset Fashion-MNIST
    def load_fashion_mnist(self):
        print("Cargando Fashion-MNIST...")
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        self.datasets['fashion'] = {
            'data': fashion_mnist.data,
            'target': fashion_mnist.target.astype(int),
            'target_names': np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
                                    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']),
            'shape': (28, 28),
            'name': 'Fashion-MNIST'
        }
        print(f"Fashion-MNIST cargado: {fashion_mnist.data.shape[0]} imágenes")
    
        #Preprocesar los datos
    def preprocess_data(self, dataset_name, n_samples=5000):
        dataset = self.datasets[dataset_name]
        X = dataset['data']
        y = dataset['target']
        
        # Submuestreo para eficiencia computacional
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Normalización
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        
        return X_normalized, y
        
        #Aplicar PCA
    def apply_pca(self, X, n_components=50):
        print(f"Aplicando PCA con {n_components} componentes...")
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Estadísticas de PCA
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        return {
            'transformed': X_pca,
            'model': pca,
            'explained_variance': explained_variance,
            'components': pca.components_
        }
        #Aplciar NMF    
    def apply_nmf(self, X, n_components=20):
        print(f"Aplicando NMF con {n_components} componentes...")
        
        # Asegurar que los datos sean positivos para NMF
        X_positive = X - np.min(X) + 1e-8
        
        nmf = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=500)
        X_nmf = nmf.fit_transform(X_positive)
        
        return {
            'transformed': X_nmf,
            'model': nmf,
            'components': nmf.components_
        }
    
    def apply_tsne(self, X, n_components=2):
        """Aplicar t-SNE"""
        print("Aplicando t-SNE...")
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        return X_tsne
    
    def apply_clustering(self, X, n_clusters=10):
        """Aplicar K-means clustering"""
        print(f"Aplicando K-means con {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels, kmeans
    
    def visualize_components(self, dataset_name, method='pca', n_components=20):
        """Visualizar componentes principales"""
        dataset = self.datasets[dataset_name]
        components = self.results[dataset_name][method]['components']
        shape = dataset['shape']
        
        # Calcular disposición de la grid
        n_cols = 5
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten()
        
        title = f"{method.upper()} Components - {dataset['name']}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(n_components):
            if i < len(components):
                if method == 'pca':
                    # Para PCA, mostrar componentes principales
                    component_img = components[i].reshape(shape)
                    # Centrar en 0 para mejor visualización
                    component_img = component_img - np.mean(component_img)
                else:
                    # Para NMF, mostrar componentes base
                    component_img = components[i].reshape(shape)
                
                axes[i].imshow(component_img, cmap='viridis')
                axes[i].set_title(f'Comp {i+1}')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_{method}_components.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_reduction(self, dataset_name, method='pca'):
        """Visualizar reducción dimensional"""
        dataset = self.datasets[dataset_name]
        X_reduced = self.results[dataset_name][method]['transformed']
        y = self.results[dataset_name]['y']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot coloreado por clase real
        scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                              cmap='tab10', alpha=0.6, s=10)
        ax1.set_title(f'{method.upper()} - Colores por clase real\n{dataset["name"]}')
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Scatter plot coloreado por clusters
        if f'{method}_clusters' in self.results[dataset_name]:
            labels = self.results[dataset_name][f'{method}_clusters']
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels,
                                  cmap='tab10', alpha=0.6, s=10)
            ax2.set_title(f'{method.upper()} - Colores por clusters\n{dataset["name"]}')
            ax2.set_xlabel(f'{method.upper()} 1')
            ax2.set_ylabel(f'{method.upper()} 2')
            plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_{method}_reduction.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_dataset(self, dataset_name, n_samples=5000):
        """Análisis completo de un dataset"""
        print(f"\n{'='*50}")
        print(f"ANALIZANDO: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Preprocesamiento
        X, y = self.preprocess_data(dataset_name, n_samples)
        self.results[dataset_name] = {'y': y}
        
        # Aplicar PCA
        pca_results = self.apply_pca(X)
        self.results[dataset_name]['pca'] = pca_results
        
        # Aplicar NMF
        nmf_results = self.apply_nmf(X)
        self.results[dataset_name]['nmf'] = nmf_results
        
        # Aplicar t-SNE en datos PCA (para eficiencia)
        X_tsne = self.apply_tsne(pca_results['transformed'][:, :50])
        self.results[dataset_name]['tsne'] = X_tsne
        
        # Clustering en espacio PCA
        pca_labels, pca_kmeans = self.apply_clustering(pca_results['transformed'][:, :20])
        self.results[dataset_name]['pca_clusters'] = pca_labels
        
        # Clustering en espacio NMF
        nmf_labels, nmf_kmeans = self.apply_clustering(nmf_results['transformed'])
        self.results[dataset_name]['nmf_clusters'] = nmf_labels
        
        # Visualizaciones
        self.visualize_components(dataset_name, 'pca', 15)
        self.visualize_components(dataset_name, 'nmf', 15)
        self.visualize_reduction(dataset_name, 'pca')
        
        # Visualización t-SNE
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
        plt.title(f't-SNE Visualization - {self.datasets[dataset_name]["name"]}')
        plt.colorbar(scatter)
        plt.savefig(f'{dataset_name}_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Métricas de evaluación
        self.evaluate_clustering(dataset_name)
        
        return self.results[dataset_name]
    
    def evaluate_clustering(self, dataset_name):
        """Evaluar resultados del clustering"""
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        
        y_true = self.results[dataset_name]['y']
        pca_labels = self.results[dataset_name]['pca_clusters']
        nmf_labels = self.results[dataset_name]['nmf_clusters']
        X_pca = self.results[dataset_name]['pca']['transformed'][:, :20]
        X_nmf = self.results[dataset_name]['nmf']['transformed']
        
        print("\nEVALUACIÓN DE CLUSTERING:")
        print(f"PCA - Adjusted Rand Score: {adjusted_rand_score(y_true, pca_labels):.3f}")
        print(f"NMF - Adjusted Rand Score: {adjusted_rand_score(y_true, nmf_labels):.3f}")
        print(f"PCA - Silhouette Score: {silhouette_score(X_pca, pca_labels):.3f}")
        print(f"NMF - Silhouette Score: {silhouette_score(X_nmf, nmf_labels):.3f}")
    
    def generate_comparative_analysis(self):
        """Generar análisis comparativo entre datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, dataset_name in enumerate(self.datasets.keys()):
            row = idx // 2
            col = idx % 2
            
            if dataset_name in self.results:
                X_pca = self.results[dataset_name]['pca']['transformed']
                y = self.results[dataset_name]['y']
                
                scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], 
                                               c=y, cmap='tab10', alpha=0.6, s=10)
                axes[row, col].set_title(f'PCA - {self.datasets[dataset_name]["name"]}')
                axes[row, col].set_xlabel('PC1')
                axes[row, col].set_ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Ejecucion principal
def main():
    processor = ImageProcessor()
    
    # Cargar datasets
    processor.load_mnist()
    processor.load_fashion_mnist()
    
    # Analizar cada dataset
    mnist_results = processor.analyze_dataset('mnist', n_samples=3000)
    fashion_results = processor.analyze_dataset('fashion', n_samples=3000)
    
    # Análisis comparativo
    processor.generate_comparative_analysis()
    
    return processor

if __name__ == "__main__":
    processor = main()

