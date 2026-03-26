import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import mlflow
import mlflow.sklearn

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_experiment(self, model_name: str, model_pipeline, data, y_test, y_pred, y_proba=None):
        """Registra o modelo, dados, métricas padrão e as métricas hierárquicas customizadas."""
        
        with mlflow.start_run(run_name=model_name):
            
            if isinstance(y_test, pd.DataFrame):
                target_names = y_test.columns.tolist() # Ex: ['y2', 'y3', 'y4']
                y_test_arr = y_test.to_numpy()
            else:
                target_names = [f"target_{i}" for i in range(y_test.shape[1])]
                y_test_arr = np.array(y_test)
                
            y_pred_arr = np.array(y_pred)

            metrics = {}

            #####################################
            # (Hierarchical/Cascading Penalty)
            #####################################
            # Bool matrix to compare
            matches = (y_test_arr == y_pred_arr)
            
            # Type 2
            l1_correct = matches[:, 0]
            
            # Type 2 + Type 3 
            l2_correct = l1_correct & matches[:, 1]
            
            # Type 2 + Type 3 + Type 4

            l3_correct = l2_correct & matches[:, 2]
            

            metrics["acc_Type2_singular"] = np.mean(l1_correct)
            metrics["acc_Type2_and_Type3"] = np.mean(l2_correct)
            metrics["acc_Type2_Type3_and_Type4"] = np.mean(l3_correct)
            
            acuracias_por_instancia = (l1_correct.astype(int) + l2_correct.astype(int) + l3_correct.astype(int)) / 3.0
            metrics["professor_cascading_accuracy_mean"] = np.mean(acuracias_por_instancia)
            # ==========================================


            for i, col_name in enumerate(target_names):
                y_t = y_test_arr[:, i] 
                y_p = y_pred_arr[:, i] 
                
                metrics[f"accuracy_raw_{col_name}"] = accuracy_score(y_t, y_p)
                metrics[f"recall_weighted_{col_name}"] = recall_score(y_t, y_p, average='weighted', zero_division=0)
                
                fig_cm = self._plot_matrix_confusion(y_t, y_p, title=f"Confusion Matrix - {col_name}")
                mlflow.log_figure(fig_cm, f"plots/confusion_matrix_{col_name}.png")
                plt.close(fig_cm)

            mlflow.log_metrics(metrics)


            X_train = data.get_X_train()
            if isinstance(X_train, pd.DataFrame):
                X_train.to_csv("X_train.csv", index=False)
            else:
                pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
                
            mlflow.log_artifact("X_train.csv", "data_versions")
            

            mlflow.sklearn.log_model(model_pipeline, "model")
            
            print(f"[{model_name}] Mlflow Saved")

    def _plot_matrix_confusion(self, y_test, y_pred, title="Confusion Matrix", cmap_="viridis"):
        cm = confusion_matrix(y_test, y_pred)
        cmd = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        cmd.plot(cmap=cmap_, ax=ax)
        ax.set_title(title)
        return fig

    def _plot_roc_curve(self, y_test, y_proba, pos_label=1):
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return fig