from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy.orm import Session
from src.models.ml_model import MLModel
from src.models.dataset import Dataset

class DashboardMetrics:
    def __init__(self, user_id: int, db: Session):
        self.user_id = user_id
        self.db = db
        
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get overall model metrics for the user"""
        models = self.db.query(MLModel).filter(
            MLModel.user_id == self.user_id
        ).all()
        
        metrics = {
            'total_models': len(models),
            'model_types': {},
            'algorithms': {},
            'status': {},
            'performance': []
        }
        
        for model in models:
            # Count model types
            metrics['model_types'][model.type] = metrics['model_types'].get(model.type, 0) + 1
            
            # Count algorithms
            metrics['algorithms'][model.algorithm] = metrics['algorithms'].get(model.algorithm, 0) + 1
            
            # Count status
            metrics['status'][model.status] = metrics['status'].get(model.status, 0) + 1
            
            # Collect performance metrics
            if model.metrics:
                metrics['performance'].append({
                    'model_name': model.name,
                    'type': model.type,
                    'algorithm': model.algorithm,
                    'metrics': model.metrics
                })
        
        return metrics
    
    def get_dataset_metrics(self) -> Dict[str, Any]:
        """Get dataset usage metrics"""
        datasets = self.db.query(Dataset).filter(
            Dataset.user_id == self.user_id
        ).all()
        
        metrics = {
            'total_datasets': len(datasets),
            'total_rows': 0,
            'total_columns': 0,
            'file_types': {},
            'feature_types': {},
            'missing_values': {}
        }
        
        for dataset in datasets:
            metrics['total_rows'] += dataset.statistics.get('rows', 0)
            metrics['total_columns'] += dataset.statistics.get('columns', 0)
            
            # Count file types
            metrics['file_types'][dataset.file_type] = metrics['file_types'].get(dataset.file_type, 0) + 1
            
            # Analyze features
            if dataset.features:
                for feature in dataset.features:
                    feature_type = dataset.statistics.get('feature_types', {}).get(feature, 'unknown')
                    metrics['feature_types'][feature_type] = metrics['feature_types'].get(feature_type, 0) + 1
            
            # Sum up missing values
            if dataset.statistics.get('missing_values'):
                for col, count in dataset.statistics['missing_values'].items():
                    metrics['missing_values'][col] = metrics['missing_values'].get(col, 0) + count
        
        return metrics
    
    def create_performance_plots(self) -> Dict[str, go.Figure]:
        """Create performance visualization plots"""
        metrics = self.get_model_metrics()
        plots = {}
        
        # Model distribution pie chart
        plots['model_distribution'] = go.Figure(data=[
            go.Pie(
                labels=list(metrics['model_types'].keys()),
                values=list(metrics['model_types'].values()),
                hole=.3
            )
        ])
        plots['model_distribution'].update_layout(
            title='Model Type Distribution',
            showlegend=True
        )
        
        # Algorithm usage bar chart
        plots['algorithm_usage'] = go.Figure(data=[
            go.Bar(
                x=list(metrics['algorithms'].keys()),
                y=list(metrics['algorithms'].values())
            )
        ])
        plots['algorithm_usage'].update_layout(
            title='Algorithm Usage',
            xaxis_title='Algorithm',
            yaxis_title='Count'
        )
        
        # Performance metrics comparison
        if metrics['performance']:
            performance_df = pd.DataFrame(metrics['performance'])
            
            # Create subplot for different metric types
            if 'classification' in metrics['model_types']:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score')
                )
                
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    row = 1 if metric in ['accuracy', 'precision'] else 2
                    col = 1 if metric in ['accuracy', 'recall'] else 2
                    
                    fig.add_trace(
                        go.Box(
                            y=[m['metrics'][metric] for m in metrics['performance']],
                            name=metric
                        ),
                        row=row, col=col
                    )
                
                plots['classification_metrics'] = fig
            
            if 'regression' in metrics['model_types']:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('MSE', 'RMSE', 'MAE', 'R2 Score')
                )
                
                for i, metric in enumerate(['mse', 'rmse', 'mae', 'r2']):
                    row = 1 if i < 2 else 2
                    col = 1 if i % 2 == 0 else 2
                    
                    fig.add_trace(
                        go.Box(
                            y=[m['metrics'][metric] for m in metrics['performance']],
                            name=metric
                        ),
                        row=row, col=col
                    )
                
                plots['regression_metrics'] = fig
        
        return plots
    
    def create_dataset_plots(self) -> Dict[str, go.Figure]:
        """Create dataset visualization plots"""
        metrics = self.get_dataset_metrics()
        plots = {}
        
        # File type distribution
        plots['file_types'] = go.Figure(data=[
            go.Pie(
                labels=list(metrics['file_types'].keys()),
                values=list(metrics['file_types'].values()),
                hole=.3
            )
        ])
        plots['file_types'].update_layout(
            title='Dataset File Type Distribution'
        )
        
        # Feature type distribution
        plots['feature_types'] = go.Figure(data=[
            go.Bar(
                x=list(metrics['feature_types'].keys()),
                y=list(metrics['feature_types'].values())
            )
        ])
        plots['feature_types'].update_layout(
            title='Feature Type Distribution',
            xaxis_title='Feature Type',
            yaxis_title='Count'
        )
        
        # Missing values heatmap
        if metrics['missing_values']:
            missing_df = pd.DataFrame.from_dict(
                metrics['missing_values'],
                orient='index',
                columns=['count']
            )
            
            plots['missing_values'] = go.Figure(data=[
                go.Heatmap(
                    z=[missing_df['count']],
                    x=missing_df.index,
                    colorscale='Viridis'
                )
            ])
            plots['missing_values'].update_layout(
                title='Missing Values Distribution',
                xaxis_title='Features',
                yaxis_title='Dataset'
            )
        
        return plots
