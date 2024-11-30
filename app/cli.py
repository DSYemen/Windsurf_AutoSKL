import click
import pandas as pd
from pathlib import Path
import json
from typing import Optional
from app.services.data_processor import DataProcessor
from app.services.model_trainer import ModelTrainer
from app.services.model_monitor import ModelMonitor
from app.services.validator import DataValidator
from app.services.report_generator import ReportGenerator

@click.group()
def cli():
    """AutoSKL Command Line Interface"""
    pass

@cli.command()
@click.option('--data', required=True, help='Path to training data file')
@click.option('--target', required=True, help='Target column name')
@click.option('--model-types', multiple=True, help='Model types to try')
@click.option('--n-trials', default=100, help='Number of optimization trials')
@click.option('--output', default='models', help='Output directory for model')
def train(
    data: str,
    target: str,
    model_types: tuple,
    n_trials: int,
    output: str
):
    """Train a new model"""
    try:
        # Load and validate data
        df = pd.read_csv(data)
        validator = DataValidator()
        validation_report = validator.validate_training_data(df, target)
        
        if validation_report.validation_errors:
            click.echo("Validation errors found:")
            for error in validation_report.validation_errors:
                click.echo(f"- {error}")
            if not click.confirm("Do you want to continue anyway?"):
                return
                
        # Process data
        processor = DataProcessor()
        X, y = processor.fit_transform(df, target)
        
        # Train model
        trainer = ModelTrainer()
        results = trainer.train(
            X, y,
            model_types=list(model_types) if model_types else None,
            n_trials=n_trials
        )
        
        # Save model
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / "model.joblib"
        trainer.save_model(str(model_path))
        
        # Generate report
        report_generator = ReportGenerator()
        feature_importance = {
            f"feature_{i}": imp
            for i, imp in enumerate(trainer.best_model.feature_importances_)
        }
        performance_metrics = {"score": results["score"]}
        report_path = report_generator.generate_training_report(
            results,
            feature_importance,
            performance_metrics,
            {}
        )
        
        click.echo(f"Model trained successfully! Score: {results['score']:.4f}")
        click.echo(f"Model saved to: {model_path}")
        click.echo(f"Training report: {report_path}")
        
    except Exception as e:
        click.echo(f"Error during training: {str(e)}", err=True)

@cli.command()
@click.option('--data', required=True, help='Path to input data file')
@click.option('--model', default='models/model.joblib', help='Path to model file')
@click.option('--output', help='Path to save predictions')
def predict(data: str, model: str, output: Optional[str]):
    """Make predictions using trained model"""
    try:
        # Load data
        df = pd.read_csv(data)
        
        # Load model
        trainer = ModelTrainer.load_model(model)
        
        # Transform data
        processor = DataProcessor()
        X = processor.transform(df)
        
        # Make predictions
        predictions = trainer.predict(X)
        
        # Save or display predictions
        if output:
            pd.DataFrame({'prediction': predictions}).to_csv(output, index=False)
            click.echo(f"Predictions saved to: {output}")
        else:
            click.echo("Predictions:")
            for i, pred in enumerate(predictions):
                click.echo(f"{i+1}: {pred}")
                
    except Exception as e:
        click.echo(f"Error during prediction: {str(e)}", err=True)

@cli.command()
@click.option('--model-id', default='latest', help='Model ID or "latest"')
@click.option('--type', type=click.Choice(['training', 'monitoring']),
              default='training', help='Report type')
@click.option('--output', help='Output directory for report')
def report(model_id: str, type: str, output: Optional[str]):
    """Generate model report"""
    try:
        # Set up paths
        model_dir = Path("models")
        if model_id == "latest":
            model_path = model_dir / "model.joblib"
        else:
            model_path = model_dir / f"model_{model_id}.joblib"
            
        if not model_path.exists():
            click.echo(f"Model not found: {model_path}", err=True)
            return
            
        # Load model and generate report
        trainer = ModelTrainer.load_model(str(model_path))
        report_generator = ReportGenerator(output_dir=output or "reports")
        
        if type == "training":
            # Generate training report
            feature_importance = {
                f"feature_{i}": imp
                for i, imp in enumerate(trainer.best_model.feature_importances_)
            }
            performance_metrics = {"score": trainer.best_score}
            report_path = report_generator.generate_training_report(
                {"model_type": "unknown", "score": trainer.best_score},
                feature_importance,
                performance_metrics,
                {}
            )
        else:
            # Generate monitoring report
            monitor = ModelMonitor()
            drift_analysis = {
                "drift_detected": False,
                "feature_drift_scores": []
            }
            performance_analysis = {
                "current_score": trainer.best_score,
                "performance_drop": 0,
                "requires_retraining": False
            }
            report_path = report_generator.generate_monitoring_report(
                drift_analysis,
                performance_analysis,
                {"predictions": []}
            )
            
        click.echo(f"Report generated: {report_path}")
        
    except Exception as e:
        click.echo(f"Error generating report: {str(e)}", err=True)

@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the API server"""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload
    )

@cli.command()
@click.option('--port', default=8501, help='Dashboard port')
def dashboard(port: int):
    """Start the dashboard"""
    import streamlit.cli
    import sys
    sys.argv = [
        "streamlit",
        "run",
        "app/ui/dashboard.py",
        "--server.port",
        str(port)
    ]
    streamlit.cli.main()

if __name__ == '__main__':
    cli()
