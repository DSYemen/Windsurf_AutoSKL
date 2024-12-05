from typing import Dict, List, Optional, Union, Any
import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sqlalchemy.orm import Session
from src.ml.serving import ModelServer
from src.ml.monitoring import ModelMonitor
from src.models.batch_job import BatchJob

class BatchProcessor:
    def __init__(
        self,
        db_session: Session,
        model_server: ModelServer,
        model_monitor: ModelMonitor,
        num_workers: int = -1,
        batch_size: int = 1000
    ):
        self.db = db_session
        self.model_server = model_server
        self.model_monitor = model_monitor
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.batch_size = batch_size
        
    def _process_batch(
        self,
        model_id: int,
        version: str,
        batch_data: pd.DataFrame,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """Process a single batch of data"""
        start_time = time.time()
        
        try:
            # Make predictions
            predictions = self.model_server.predict(
                model_id,
                batch_data.to_dict(orient='records')[0],
                version=version,
                return_proba=return_proba
            )
            
            status = 'success'
        except Exception as e:
            predictions = str(e)
            status = 'error'
            
        latency = time.time() - start_time
        
        # Log predictions
        for i, row in batch_data.iterrows():
            self.model_monitor.log_prediction(
                model_id=model_id,
                version=version,
                input_data=row.to_dict(),
                prediction=predictions,
                latency=latency,
                status=status
            )
            
        return {
            'predictions': predictions,
            'status': status,
            'latency': latency
        }
        
    async def process_batch_job(
        self,
        job_id: int,
        data: Union[pd.DataFrame, str],
        model_id: int,
        version: Optional[str] = None,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """Process a batch job"""
        # Update job status
        job = self.db.query(BatchJob).filter_by(id=job_id).first()
        if not job:
            raise ValueError(f"Batch job {job_id} not found")
            
        job.status = 'processing'
        job.started_at = datetime.utcnow()
        self.db.commit()
        
        try:
            # Load data if path provided
            if isinstance(data, str):
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                else:
                    raise ValueError("Unsupported file format")
            else:
                df = data
                
            # Split data into batches
            n_samples = len(df)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            batches = [
                df.iloc[i * self.batch_size:(i + 1) * self.batch_size]
                for i in range(n_batches)
            ]
            
            # Process batches in parallel
            results = []
            errors = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_batch = {
                    executor.submit(
                        self._process_batch,
                        model_id,
                        version,
                        batch,
                        return_proba
                    ): i
                    for i, batch in enumerate(batches)
                }
                
                for future in future_to_batch:
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            results.append(result)
                        else:
                            errors.append({
                                'batch': batch_idx,
                                'error': result['predictions']
                            })
                    except Exception as e:
                        errors.append({
                            'batch': batch_idx,
                            'error': str(e)
                        })
                        
            # Calculate statistics
            total_predictions = len(results)
            successful_predictions = len([r for r in results if r['status'] == 'success'])
            failed_predictions = len(errors)
            
            avg_latency = np.mean([r['latency'] for r in results]) if results else 0
            
            # Update job status
            job.status = 'completed'
            job.completed_at = datetime.utcnow()
            job.results = {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'average_latency': avg_latency,
                'errors': errors
            }
            self.db.commit()
            
            return job.results
            
        except Exception as e:
            # Update job status on error
            job.status = 'failed'
            job.completed_at = datetime.utcnow()
            job.results = {'error': str(e)}
            self.db.commit()
            raise
            
    async def get_job_status(self, job_id: int) -> Dict[str, Any]:
        """Get status of a batch job"""
        job = self.db.query(BatchJob).filter_by(id=job_id).first()
        if not job:
            raise ValueError(f"Batch job {job_id} not found")
            
        return {
            'id': job.id,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'results': job.results
        }
        
    def cancel_job(self, job_id: int) -> None:
        """Cancel a batch job"""
        job = self.db.query(BatchJob).filter_by(id=job_id).first()
        if not job:
            raise ValueError(f"Batch job {job_id} not found")
            
        if job.status in ['queued', 'processing']:
            job.status = 'cancelled'
            job.completed_at = datetime.utcnow()
            self.db.commit()
            
    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old completed jobs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        deleted = self.db.query(BatchJob).filter(
            BatchJob.status.in_(['completed', 'failed', 'cancelled']),
            BatchJob.completed_at <= cutoff_date
        ).delete()
        
        self.db.commit()
        return deleted
