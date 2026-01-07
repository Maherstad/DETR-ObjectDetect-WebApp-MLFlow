"""
Hyperparameter Optimization for DETR Object Detection
Uses Optuna for hyperparameter search with MLflow tracking
"""

import optuna
import mlflow
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments
from datetime import datetime
import logging
from typing import Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DETRHyperparameterOptimizer:
    """
    Hyperparameter optimization for DETR model using Optuna
    Tracks all trials in MLflow for full experiment visibility
    """
    
    def __init__(self,
                 model_name="facebook/detr-resnet-50",
                 tracking_uri="http://127.0.0.1:5050",
                 experiment_name="DETR_Hyperparameter_Optimization",
                 n_trials=20,
                 timeout_hours=None):
        
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.timeout_hours = timeout_hours
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization
        Returns validation metric to optimize
        """
        
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'num_epochs': trial.suggest_int('num_epochs', 5, 30),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 1000),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['linear', 'cosine', 'constant']),
            'dropout': trial.suggest_uniform('dropout', 0.0, 0.3),
        }
        
        # Start MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log all hyperparameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("device", str(self.device))
            
            try:
                # Load model
                processor = DetrImageProcessor.from_pretrained(self.model_name, revision="no_timm")
                model = DetrForObjectDetection.from_pretrained(self.model_name, revision="no_timm")
                model.to(self.device)
                
                # Simulate training with these hyperparameters
                # In production, replace with actual training loop
                val_map = self._simulate_training(model, params, trial)
                
                # Log metrics
                mlflow.log_metric("val_mAP", val_map)
                mlflow.log_metric("final_metric", val_map)
                
                # Tag the run
                mlflow.set_tag("optuna_trial", trial.number)
                mlflow.set_tag("status", "completed")
                
                logger.info(f"Trial {trial.number} completed with mAP: {val_map:.4f}")
                
                return val_map
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
                raise optuna.TrialPruned()
    
    def _simulate_training(self, model, params: Dict[str, Any], trial: optuna.Trial) -> float:
        """
        Simulate model training with given hyperparameters
        In production, replace with actual training code
        """
        logger.info(f"Training with params: LR={params['learning_rate']:.2e}, BS={params['batch_size']}, Epochs={params['num_epochs']}")
        
        # Simulate training loop
        best_val_map = 0.0
        for epoch in range(params['num_epochs']):
            # Simulated metrics based on hyperparameters
            # Better learning rates and batch sizes lead to better performance
            epoch_map = 0.3 + (epoch / params['num_epochs']) * 0.4
            epoch_map += (params['learning_rate'] * 100000) * 0.1  # Reward good LRs
            epoch_map += (1.0 / params['batch_size']) * 0.05  # Smaller batches slightly better
            epoch_map += (1.0 - params['dropout']) * 0.1  # Less dropout slightly better
            
            # Add some randomness
            import random
            epoch_map += random.uniform(-0.05, 0.05)
            epoch_map = min(0.95, max(0.2, epoch_map))  # Clamp between 0.2 and 0.95
            
            best_val_map = max(best_val_map, epoch_map)
            
            # Report intermediate value for pruning
            trial.report(epoch_map, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        return best_val_map
    
    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization
        Returns the Optuna study object with all trial results
        """
        logger.info("=" * 70)
        logger.info("Starting Hyperparameter Optimization")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Device: {self.device}")
        logger.info(f"MLflow Tracking URI: {self.tracking_uri}")
        logger.info("=" * 70)
        
        # Create parent run for the optimization study
        with mlflow.start_run(run_name=f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tag("optimization_framework", "optuna")
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("timeout_hours", self.timeout_hours)
            
            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",  # Maximize mAP
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=5,
                ),
                study_name=f"detr_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Run optimization
            timeout_seconds = self.timeout_hours * 3600 if self.timeout_hours else None
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=timeout_seconds,
                show_progress_bar=True
            )
            
            # Log best results
            logger.info("\n" + "=" * 70)
            logger.info("Optimization Complete!")
            logger.info("=" * 70)
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best mAP: {study.best_value:.4f}")
            logger.info("\nBest hyperparameters:")
            for key, value in study.best_params.items():
                logger.info(f"  {key}: {value}")
                mlflow.log_param(f"best_{key}", value)
            
            mlflow.log_metric("best_mAP", study.best_value)
            mlflow.log_metric("n_completed_trials", len(study.trials))
            
            # Save optimization history
            history_df = study.trials_dataframe()
            history_file = "optimization_history.csv"
            history_df.to_csv(history_file, index=False)
            mlflow.log_artifact(history_file)
            
            # Save best parameters as JSON
            best_params_file = "best_hyperparameters.json"
            with open(best_params_file, 'w') as f:
                json.dump(study.best_params, f, indent=2)
            mlflow.log_artifact(best_params_file)
            
            # Create and log visualization
            try:
                import plotly
                # Optimization history plot
                fig1 = optuna.visualization.plot_optimization_history(study)
                fig1.write_html("optimization_history.html")
                mlflow.log_artifact("optimization_history.html")
                
                # Hyperparameter importance
                fig2 = optuna.visualization.plot_param_importances(study)
                fig2.write_html("param_importances.html")
                mlflow.log_artifact("param_importances.html")
                
                logger.info("Optimization visualizations saved")
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")
            
            logger.info("=" * 70 + "\n")
            
        return study
    
    def run_with_best_params(self, study: optuna.Study):
        """
        Train final model with best hyperparameters found
        """
        logger.info("Training final model with best hyperparameters...")
        
        with mlflow.start_run(run_name=f"final_model_best_params"):
            # Log that this is the final model
            mlflow.set_tag("model_type", "final_optimized")
            mlflow.set_tag("optimization_study", study.study_name)
            
            # Log best parameters
            for key, value in study.best_params.items():
                mlflow.log_param(key, value)
            
            # Train with best parameters (implement actual training)
            # This is a placeholder
            logger.info("Training final model...")
            final_map = study.best_value + 0.02  # Simulated improvement
            
            mlflow.log_metric("final_mAP", final_map)
            mlflow.log_metric("optimization_improvement", final_map - 0.42)
            
            logger.info(f"Final model mAP: {final_map:.4f}")
            
            return final_map


def main():
    """Main entry point for hyperparameter optimization"""
    optimizer = DETRHyperparameterOptimizer(
        n_trials=20,
        timeout_hours=None  # No timeout
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Train final model with best parameters
    final_map = optimizer.run_with_best_params(study)
    
    return study, final_map


if __name__ == "__main__":
    main()
