from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import json
from pathlib import Path
from typing import Optional

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_training_report(
        self,
        model_results: Dict[str, Any],
        feature_importance: Dict[str, float],
        performance_metrics: Dict[str, float],
        validation_results: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"training_report_{timestamp}.html"
        
        # Create HTML report
        html_content = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "</style>",
            "</head><body>",
            "<h1>Model Training Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Model Information Section
        html_content.extend([
            "<div class='section'>",
            "<h2>Model Information</h2>",
            "<table>",
            f"<tr><td>Best Model Type</td><td>{model_results['model_type']}</td></tr>",
            f"<tr><td>Model Score</td><td>{model_results['score']:.4f}</td></tr>",
            "</table>",
            "</div>"
        ])
        
        # Feature Importance Plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(feature_importance.keys()),
                y=list(feature_importance.values())
            )
        ])
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance Score"
        )
        html_content.append(fig.to_html(full_html=False))
        
        # Performance Metrics
        html_content.extend([
            "<div class='section'>",
            "<h2>Performance Metrics</h2>",
            "<table>",
            *[f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
              for k, v in performance_metrics.items()],
            "</table>",
            "</div>"
        ])
        
        # Validation Results
        if validation_results.get('confusion_matrix') is not None:
            cm = validation_results['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Confusion Matrix"
            )
            html_content.append(fig.to_html(full_html=False))
        
        html_content.append("</body></html>")
        
        # Save report
        report_path.write_text("\n".join(html_content))
        return str(report_path)
        
    def generate_monitoring_report(
        self,
        drift_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        predictions_analysis: Dict[str, Any]
    ) -> str:
        """Generate a monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"monitoring_report_{timestamp}.html"
        
        html_content = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }",
            ".alert { color: red; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "</style>",
            "</head><body>",
            "<h1>Model Monitoring Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Drift Analysis Section
        html_content.extend([
            "<div class='section'>",
            "<h2>Data Drift Analysis</h2>",
            f"<p class='{'alert' if drift_analysis['drift_detected'] else ''}'>",
            f"Drift Detected: {drift_analysis['drift_detected']}</p>",
            "<h3>Feature Drift Scores</h3>",
            "<table>",
            "<tr><th>Feature</th><th>Statistic</th><th>P-Value</th><th>Has Drift</th></tr>"
        ])
        
        for score in drift_analysis['feature_drift_scores']:
            html_content.append(
                f"<tr><td>{score['feature_index']}</td>"
                f"<td>{score['statistic']:.4f}</td>"
                f"<td>{score['p_value']:.4f}</td>"
                f"<td>{'Yes' if score['has_drift'] else 'No'}</td></tr>"
            )
        
        html_content.append("</table></div>")
        
        # Performance Analysis Section
        html_content.extend([
            "<div class='section'>",
            "<h2>Performance Analysis</h2>",
            "<table>",
            f"<tr><td>Current Score</td><td>{performance_analysis['current_score']:.4f}</td></tr>",
            f"<tr><td>Performance Drop</td><td>{performance_analysis['performance_drop']:.4f}</td></tr>",
            f"<tr><td>Requires Retraining</td><td>{performance_analysis['requires_retraining']}</td></tr>",
            "</table>",
            "</div>"
        ])
        
        # Predictions Analysis
        if predictions_analysis:
            fig = go.Figure(data=[
                go.Histogram(x=predictions_analysis['predictions'])
            ])
            fig.update_layout(
                title="Prediction Distribution",
                xaxis_title="Predicted Values",
                yaxis_title="Count"
            )
            html_content.append(fig.to_html(full_html=False))
        
        html_content.append("</body></html>")
        
        # Save report
        report_path.write_text("\n".join(html_content))
        return str(report_path)
        
    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Plot feature importance using plotly
        
        Args:
            feature_importance: Dictionary mapping feature names to their importance scores
            
        Returns:
            plotly.graph_objects.Figure: Feature importance bar plot
        """
        if not feature_importance:
            # Return empty figure if no feature importance data
            fig = go.Figure()
            fig.add_annotation(
                text="لا توجد بيانات لأهمية المتغيرات",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig
            
        # Convert to dataframe and sort by importance
        df = pd.DataFrame([
            {"المتغير": feature, "الأهمية": importance}
            for feature, importance in feature_importance.items()
        ]).sort_values("الأهمية", ascending=True)
        
        # Create horizontal bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["الأهمية"],
            y=df["المتغير"],
            orientation='h'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "أهمية المتغيرات",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="درجة الأهمية",
            yaxis_title="المتغير",
            height=max(400, len(feature_importance) * 25),  # Dynamic height based on number of features
            showlegend=False,
            font=dict(size=12),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Add percentage labels
        total_importance = df["الأهمية"].sum()
        percentages = (df["الأهمية"] / total_importance * 100).round(1)
        
        for i, (imp, pct) in enumerate(zip(df["الأهمية"], percentages)):
            fig.add_annotation(
                x=imp,
                y=i,
                text=f"{pct}%",
                xanchor='left',
                showarrow=False,
                font=dict(size=10),
                xshift=5
            )
        
        return fig
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: Optional[List[str]] = None) -> go.Figure:
        """Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional list of class labels
            
        Returns:
            plotly.graph_objects.Figure: Confusion matrix heatmap
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = [str(i) for i in range(len(cm))]
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='RdBu',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "مصفوفة الالتباس",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="التنبؤات",
            yaxis_title="القيم الحقيقية",
            width=500,
            height=500,
            font=dict(size=12)
        )
        
        return fig
        
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                      pos_label: Optional[int] = None) -> go.Figure:
        """Plot ROC curve
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            pos_label: Label of positive class
            
        Returns:
            plotly.graph_objects.Figure: ROC curve plot
        """
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})'
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "منحنى ROC",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="معدل الإيجابيات الخاطئة",
            yaxis_title="معدل الإيجابيات الصحيحة",
            width=600,
            height=500,
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig

    def save_json_report(self, data: Dict[str, Any], report_name: str):
        """Save report data as JSON"""
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(report_path)
