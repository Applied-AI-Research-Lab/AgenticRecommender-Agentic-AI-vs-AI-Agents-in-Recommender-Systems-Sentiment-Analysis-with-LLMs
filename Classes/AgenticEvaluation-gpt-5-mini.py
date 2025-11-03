import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

class AgenticEvaluation:
    def __init__(self, csv_path: str):
        """Load dataset and initialize model/orchestrator lists."""
        self.df = pd.read_csv(csv_path)
        self.models = [
            'gemini-2.5-pro_prediction',
            'gemini-2.5-flash_prediction',
            'gemini-2.5-flash-lite_prediction',
            'gpt-5_prediction',
            'gpt-5-mini_prediction',
            'gpt-5-nano-2025-08-07_prediction',
            'gpt-4.1_prediction',
            'claude-sonnet-4-5_prediction',
            'claude-haiku-4-5_prediction',
            'claude-opus-4-1_prediction',
            'deepseek-chat_prediction',
            'deepseek-reasoner_prediction'
        ]
        self.orchestrator = 'gpt-5-mini-2025-08-07_agentic_prediction'
        self.true_label = 'rating'
        # Round ratings and predictions to nearest int for classification
        self.df[self.true_label] = self.df[self.true_label].round().astype(int)
        for col in self.models + [self.orchestrator]:
            self.df[col] = self.df[col].round().astype(int)
        
        # Costs in USD for the whole dataset
        self.costs = {
            'gemini-2.5-pro_prediction': 43.97,
            'gemini-2.5-flash_prediction': 7.36,
            'gemini-2.5-flash-lite_prediction': 0.38,
            'gpt-5_prediction': 11.33,
            'gpt-5-mini_prediction': 1.69,
            'gpt-5-nano-2025-08-07_prediction': 0.67,
            'gpt-4.1_prediction': 3.09,
            'claude-sonnet-4-5_prediction': 5.76,
            'claude-haiku-4-5_prediction': 1.92,
            'claude-opus-4-1_prediction': 27.33,
            'deepseek-chat_prediction': 0.24,
            'deepseek-reasoner_prediction': 1.98,
            'gpt-5-2025-08-07_agentic_prediction': 24.73,
            'gpt-5-mini-2025-08-07_agentic_prediction': 6.14
        }

    def compute_model_metrics(self) -> pd.DataFrame:
        """Compute accuracy, precision, recall, and F1-score for each model.
        
        Returns:
            pd.DataFrame: A DataFrame with columns ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'] 
            for each model in self.models plus the orchestrator. This provides a comprehensive comparison 
            of individual model performances and the orchestrator's overall metrics as required for 
            evaluating multiple AI models.
        """
        metrics = []
        for model in self.models + [self.orchestrator]:
            y_true = self.df[self.true_label]
            y_pred = self.df[model]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics.append({
                'Model': model,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })
        return pd.DataFrame(metrics)

    def label_confusion(self, model_name: str):
        """Return confusion matrix and identify weak label areas.
        
        Args:
            model_name (str): The name of the model to analyze (e.g., 'gpt-5_prediction').
        
        Returns:
            tuple: (confusion_matrix, weak_labels)
            - confusion_matrix (np.ndarray): 5x5 matrix showing true vs predicted labels (1-5).
            - weak_labels (list): List of tuples [(label, recall_score)] for the 2 labels with lowest recall, 
              indicating which ratings (1-5) the model struggles with the most.
        """
        y_true = self.df[self.true_label]
        y_pred = self.df[model_name]
        cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])
        # Identify weak labels: labels with lowest recall
        report = classification_report(y_true, y_pred, labels=[1,2,3,4,5], output_dict=True, zero_division=0)
        weak_labels = sorted([(label, report[str(label)]['recall']) for label in [1,2,3,4,5]], key=lambda x: x[1])[:2]
        return cm, weak_labels

    def compare_orchestrator_vs_gpt5(self):
        """Compare orchestrator predictions with GPT-5-mini baseline.
        
        Returns:
            dict: Contains:
            - 'Total Changes': Number of times the orchestrator changed the GPT-5-mini prediction.
            - 'Improved Accuracy': Number of changes that led to correct predictions (when GPT-5-mini was wrong).
            - 'Worsened Accuracy': Number of changes that led to incorrect predictions (when GPT-5-mini was right).
            This analyzes how the orchestrator modifies the standalone GPT-5-mini model.
        """
        gpt5 = 'gpt-5-mini_prediction'
        orch = self.orchestrator
        y_true = self.df[self.true_label]
        changes = (self.df[gpt5] != self.df[orch]).sum()
        improved = 0
        worsened = 0
        for idx in self.df.index:
            if self.df.loc[idx, gpt5] != self.df.loc[idx, orch]:
                gpt_correct = self.df.loc[idx, gpt5] == y_true[idx]
                orch_correct = self.df.loc[idx, orch] == y_true[idx]
                if not gpt_correct and orch_correct:
                    improved += 1
                elif gpt_correct and not orch_correct:
                    worsened += 1
        return {
            'Total Changes': changes,
            'Improved Accuracy': improved,
            'Worsened Accuracy': worsened
        }

    def model_agreement_with_orchestrator(self) -> pd.DataFrame:
        """Compute how often orchestrator agrees with each model.
        
        Returns:
            pd.DataFrame: Columns ['Model', 'Agreement Count', 'Agreement Percentage'].
            Shows the count and percentage of predictions where each individual model agrees with the orchestrator,
            helping identify which model the orchestrator aligns most closely with.
        """
        agreements = []
        for model in self.models:
            agree = (self.df[model] == self.df[self.orchestrator]).sum()
            total = len(self.df)
            agreements.append({
                'Model': model,
                'Agreement Count': agree,
                'Agreement Percentage': agree / total * 100
            })
        return pd.DataFrame(agreements)

    def analyze_influences(self) -> pd.DataFrame:
        """Determine which models positively or negatively influence orchestrator decisions.
        
        Returns:
            pd.DataFrame: Columns ['Model', 'Positive Influence', 'Negative Influence'].
            For each model, counts how many times their prediction matched the orchestrator's change 
            and whether that change improved (positive) or worsened (negative) accuracy.
            Analyzes influence when the orchestrator changes from the GPT-5-mini baseline.
        """
        gpt5 = 'gpt-5-mini_prediction'
        influences = {model: {'positive': 0, 'negative': 0} for model in self.models}
        y_true = self.df[self.true_label]
        for idx in self.df.index:
            if self.df.loc[idx, gpt5] != self.df.loc[idx, self.orchestrator]:
                new_pred = self.df.loc[idx, self.orchestrator]
                correct = new_pred == y_true[idx]
                for model in self.models:
                    if self.df.loc[idx, model] == new_pred:
                        if correct:
                            influences[model]['positive'] += 1
                        else:
                            influences[model]['negative'] += 1
        # Flatten to DataFrame
        data = []
        for model, infl in influences.items():
            data.append({
                'Model': model,
                'Positive Influence': infl['positive'],
                'Negative Influence': infl['negative']
            })
        return pd.DataFrame(data)

    def system_value_analysis(self):
        """Compare orchestrator performance vs. individual model average.
        
        Returns:
            dict: Contains:
            - 'Orchestrator Accuracy': Accuracy of the orchestrator model.
            - 'Average Model Accuracy': Mean accuracy of all individual models.
            - 'Improvement': Difference (orchestrator - average), indicating if the agentic AI system 
              is more valuable than individual models.
        """
        metrics_df = self.compute_model_metrics()
        orch_acc = metrics_df[metrics_df['Model'] == self.orchestrator]['Accuracy'].values[0]
        avg_acc = metrics_df[metrics_df['Model'].isin(self.models)]['Accuracy'].mean()
        return {
            'Orchestrator Accuracy': orch_acc,
            'Average Model Accuracy': avg_acc,
            'Improvement': orch_acc - avg_acc
        }

    # Optional Enhancements
    def export_metrics_to_excel(self, output_path: str = 'Results/model_metrics.xlsx'):
        """Export model metrics to Excel.
        
        Args:
            output_path (str): Path to save the Excel file (default: 'Results/model_metrics.xlsx').
        
        Returns:
            None: Saves the DataFrame from compute_model_metrics() to an Excel file in the Results folder.
        """
        metrics_df = self.compute_model_metrics()
        metrics_df.to_excel(output_path, index=False)

    def plot_metrics_bar_chart(self, output_path: str = 'Results/metrics_bar_chart.png'):
        """Visualize metrics with bar chart.
        
        Args:
            output_path (str): Path to save the plot image (default: 'Results/metrics_bar_chart.png').
        
        Returns:
            None: Saves a bar chart comparing Accuracy, Precision, Recall, F1-Score for all models 
            and the orchestrator to the Results folder.
        """
        metrics_df = self.compute_model_metrics()
        metrics_df.set_index('Model', inplace=True)
        metrics_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, model_name: str, output_path: str = 'Results/confusion_matrix.png'):
        """Plot confusion matrix for a model.
        
        Args:
            model_name (str): The model to plot confusion matrix for.
            output_path (str): Path to save the plot (default: 'Results/confusion_matrix.png').
        
        Returns:
            None: Saves a heatmap visualization of the confusion matrix for the specified model 
            to the Results folder.
        """
        cm, _ = self.label_confusion(model_name)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path)
        plt.close()

    def display_divergent_reviews(self, n_examples: int = 5):
        """Display example reviews where predictions diverge significantly.
        
        Args:
            n_examples (int): Number of examples to return (default: 5).
        
        Returns:
            pd.DataFrame: Columns ['title', 'text', 'rating', 'gpt-5_prediction', 'orchestrator_prediction'].
            Shows reviews where the orchestrator prediction differs from GPT-5, highlighting 
            significant divergences in predictions.
        """
        # Find rows where orchestrator differs from gpt5
        divergent = self.df[self.df['gpt-5_prediction'] != self.df[self.orchestrator]]
        examples = divergent.head(n_examples)[['title', 'text', 'rating', 'gpt-5_prediction', self.orchestrator]]
        return examples

    def compute_prediction_correlation(self):
        """Compute correlation among model predictions.
        
        Returns:
            pd.DataFrame: Correlation matrix for all model predictions including the orchestrator.
            Shows how predictions from different models correlate, useful for understanding 
            similarities in model behaviors.
        """
        pred_cols = self.models + [self.orchestrator]
        corr = self.df[pred_cols].corr()
        return corr

    def majority_vote_baseline(self):
        """Implement a majority-vote baseline for comparison.
        
        Returns:
            float: Accuracy of a majority-vote ensemble using all individual models.
            Provides a baseline accuracy to compare against the orchestrator's performance.
        """
        pred_cols = self.models
        majority_preds = []
        for idx in self.df.index:
            preds = [self.df.loc[idx, col] for col in pred_cols]
            majority = Counter(preds).most_common(1)[0][0]
            majority_preds.append(majority)
        y_true = self.df[self.true_label]
        acc = accuracy_score(y_true, majority_preds)
        return acc

    # New functions for missing research questions

    def rating_failure_analysis(self):
        """Analyze for which rating levels (1-5) each AI Agent most often fails to predict correctly.
        
        Returns:
            dict: For each model, a dict of {rating: failure_percentage}, and overall hardest rating.
        """
        results = {}
        total_per_rating = self.df[self.true_label].value_counts().to_dict()
        for model in self.models:
            failures = {rating: 0 for rating in range(1, 6)}
            y_true = self.df[self.true_label]
            y_pred = self.df[model]
            for idx in self.df.index:
                if y_true[idx] != y_pred[idx]:
                    failures[y_true[idx]] += 1
            # Convert to percentages
            failure_pct = {rating: (failures[rating] / total_per_rating.get(rating, 1)) * 100 for rating in range(1, 6)}
            results[model] = failure_pct
        
        # Overall hardest rating: average failure percentage across models
        avg_failures = {rating: sum(results[model][rating] for model in self.models) / len(self.models) for rating in range(1, 6)}
        hardest_rating = max(avg_failures, key=avg_failures.get)
        
        results['Overall Hardest Rating'] = hardest_rating
        results['Average Failure Percentages'] = avg_failures
        return results

    def orchestrator_disregard_analysis(self):
        """Analyze how often the orchestrator disregards agent recommendations and if advantageous.
        
        Assumes disregarding means orchestrator prediction doesn't match any agent's prediction.
        
        Returns:
            dict: 'Disregard Count', 'Disregard Percentage', 'Correct Disregards', 'Incorrect Disregards'
        """
        disregard_count = 0
        correct_disregards = 0
        incorrect_disregards = 0
        y_true = self.df[self.true_label]
        for idx in self.df.index:
            orch_pred = self.df.loc[idx, self.orchestrator]
            agent_preds = [self.df.loc[idx, model] for model in self.models]
            if orch_pred not in agent_preds:
                disregard_count += 1
                if orch_pred == y_true[idx]:
                    correct_disregards += 1
                else:
                    incorrect_disregards += 1
        total = len(self.df)
        return {
            'Disregard Count': disregard_count,
            'Disregard Percentage': disregard_count / total * 100,
            'Correct Disregards': correct_disregards,
            'Incorrect Disregards': incorrect_disregards
        }

    def compare_orchestrator_vs_standalone(self):
        """Compare orchestrator performance to its standalone version (assuming GPT-5).
        
        Returns:
            dict: Similar to compare_orchestrator_vs_gpt5, but for standalone vs agentic.
        """
        standalone = 'gpt-5_prediction'
        orch = self.orchestrator
        y_true = self.df[self.true_label]
        changes = (self.df[standalone] != self.df[orch]).sum()
        improved = 0
        worsened = 0
        for idx in self.df.index:
            if self.df.loc[idx, standalone] != self.df.loc[idx, orch]:
                stand_correct = self.df.loc[idx, standalone] == y_true[idx]
                orch_correct = self.df.loc[idx, orch] == y_true[idx]
                if not stand_correct and orch_correct:
                    improved += 1
                elif stand_correct and not orch_correct:
                    worsened += 1
        return {
            'Total Changes': changes,
            'Improved Accuracy': improved,
            'Worsened Accuracy': worsened
        }

    def voting_vs_reasoning_analysis(self):
        """Analyze if orchestrator relies on majority voting or reasons directly.
        
        Compares orchestrator predictions to majority vote.
        
        Returns:
            dict: 'Matches Majority', 'Differs from Majority', 'Accuracy when Matches', 'Accuracy when Differs'
        """
        majority_preds = []
        for idx in self.df.index:
            preds = [self.df.loc[idx, model] for model in self.models]
            majority = Counter(preds).most_common(1)[0][0]
            majority_preds.append(majority)
        matches = 0
        differs = 0
        correct_matches = 0
        correct_differs = 0
        y_true = self.df[self.true_label]
        orch_preds = self.df[self.orchestrator].tolist()
        for i, idx in enumerate(self.df.index):
            if orch_preds[i] == majority_preds[i]:
                matches += 1
                if orch_preds[i] == y_true[idx]:
                    correct_matches += 1
            else:
                differs += 1
                if orch_preds[i] == y_true[idx]:
                    correct_differs += 1
        total = len(self.df)
        return {
            'Matches Majority': matches,
            'Differs from Majority': differs,
            'Accuracy when Matches': correct_matches / matches if matches > 0 else 0,
            'Accuracy when Differs': correct_differs / differs if differs > 0 else 0
        }

    def outlier_agents(self):
        """Identify agents that behave as outliers, potentially disrupting the system.
        
        Based on low agreement with orchestrator and high negative influence.
        
        Returns:
            list: List of outlier models.
        """
        agreement_df = self.model_agreement_with_orchestrator()
        influences_df = self.analyze_influences()
        outliers = []
        mean_agreement = agreement_df['Agreement Percentage'].mean()
        for _, row in agreement_df.iterrows():
            model = row['Model']
            agree_pct = row['Agreement Percentage']
            neg_infl = influences_df[influences_df['Model'] == model]['Negative Influence'].values[0]
            if agree_pct < mean_agreement and neg_infl > influences_df['Negative Influence'].mean():
                outliers.append(model)
        return outliers

    def pretrained_llm_capability(self):
        """Assess if pre-trained LLMs without fine-tuning can accurately capture sentiment.
        
        Returns:
            dict: 'Average Accuracy', 'Max Accuracy', 'Min Accuracy'
        """
        metrics_df = self.compute_model_metrics()
        model_metrics = metrics_df[metrics_df['Model'].isin(self.models)]
        avg_acc = model_metrics['Accuracy'].mean()
        max_acc = model_metrics['Accuracy'].max()
        min_acc = model_metrics['Accuracy'].min()
        return {
            'Average Accuracy': avg_acc,
            'Max Accuracy': max_acc,
            'Min Accuracy': min_acc
        }

    def easiest_rating_analysis(self):
        """Analyze which rating levels (1-5) are easiest for each AI Agent to predict correctly.
        
        Returns:
            dict: For each model, a dict of {rating: success_percentage}, and overall easiest rating.
        """
        results = {}
        total_per_rating = self.df[self.true_label].value_counts().to_dict()
        for model in self.models:
            successes = {rating: 0 for rating in range(1, 6)}
            y_true = self.df[self.true_label]
            y_pred = self.df[model]
            for idx in self.df.index:
                if y_true[idx] == y_pred[idx]:
                    successes[y_true[idx]] += 1
            # Convert to percentages
            success_pct = {rating: (successes[rating] / total_per_rating.get(rating, 1)) * 100 for rating in range(1, 6)}
            results[model] = success_pct
        
        # Overall easiest rating: average success percentage across models
        avg_successes = {rating: sum(results[model][rating] for model in self.models) / len(self.models) for rating in range(1, 6)}
        easiest_rating = max(avg_successes, key=avg_successes.get)
        
        results['Overall Easiest Rating'] = easiest_rating
        results['Average Success Percentages'] = avg_successes
        return results

    def cost_benefit_analysis(self):
        """Analyze if Agentic AI is worth the complexity, balancing accuracy against computational cost.
        
        Returns:
            dict: Accuracy improvement, total individual costs, orchestrator cost, cost per accuracy point.
        """
        value = self.system_value_analysis()
        total_individual_cost = sum(self.costs[model] for model in self.models)
        orch_cost = self.costs[self.orchestrator]
        avg_individual_acc = value['Average Model Accuracy']
        orch_acc = value['Orchestrator Accuracy']
        improvement = value['Improvement']
        
        # Cost per accuracy point (simplified, assuming linear)
        if improvement > 0:
            cost_per_acc_point = (orch_cost - total_individual_cost) / improvement
        else:
            cost_per_acc_point = float('inf')
        
        conclusion = "Agentic AI improves accuracy"
        if cost_per_acc_point < 0:
            conclusion += " at a lower cost."
        else:
            conclusion += " but at a higher cost."
        conclusion += " Evaluate if the improvement justifies the expense."
        
        return {
            'Accuracy Improvement': improvement,
            'Total Individual Models Cost': total_individual_cost,
            'Orchestrator Cost': orch_cost,
            'Cost Difference': orch_cost - total_individual_cost,
            'Cost per Accuracy Point Improvement': cost_per_acc_point,
            'Conclusion': conclusion
        }

    def analyze_orchestrator_reasoning(self):
        """Analyze the orchestrator's reasoning column to extract insights.
        
        Returns:
            dict: Analysis of reasoning patterns, agreement levels, model mentions, etc.
        """
        reasoning_col = self.orchestrator.replace('_prediction', '_reasoning')
        if reasoning_col not in self.df.columns:
            return {"Error": "Reasoning column not found"}
        
        reasonings = self.df[reasoning_col].dropna()
        total_cases = len(reasonings)
        
        # Count cases with unanimous agreement
        unanimous = 0
        deviations = 0
        for idx in reasonings.index:
            text = reasonings[idx]
            if "no deviations" in text.lower() or "unanimously" in text.lower() or "all agents predicted" in text.lower():
                unanimous += 1
            else:
                deviations += 1
        
        # Average reasoning length
        avg_length = reasonings.str.len().mean()
        
        # Count model mentions
        model_keywords = {
            'gemini-2.5-pro': ['gemini-2.5-pro'],
            'gemini-2.5-flash': ['gemini-2.5-flash'],
            'gemini-2.5-flash-lite': ['gemini-2.5-flash-lite'],
            'gpt-5': ['gpt-5'],
            'gpt-5-mini': ['gpt-5-mini'],
            'gpt-5-nano': ['gpt-5-nano'],
            'gpt-4.1': ['gpt-4.1'],
            'claude-sonnet': ['claude-sonnet'],
            'claude-haiku': ['claude-haiku'],
            'claude-opus': ['claude-opus'],
            'deepseek-chat': ['deepseek-chat'],
            'deepseek-reasoner': ['deepseek-reasoner']
        }
        
        model_mentions = {model: 0 for model in model_keywords}
        for idx in reasonings.index:
            text = reasonings[idx].lower()
            for model, keywords in model_keywords.items():
                for kw in keywords:
                    if kw in text:
                        model_mentions[model] += 1
                        break  # Count once per reasoning
        
        # Sort by mentions
        sorted_mentions = sorted(model_mentions.items(), key=lambda x: x[1], reverse=True)
        
        # Accuracy when unanimous vs deviations
        orch_preds = self.df.loc[reasonings.index, self.orchestrator]
        true_labels = self.df.loc[reasonings.index, self.true_label]
        correct_unanimous = 0
        correct_deviations = 0
        
        for idx in reasonings.index:
            text = reasonings[idx]
            correct = orch_preds[idx] == true_labels[idx]
            if "no deviations" in text.lower() or "unanimously" in text.lower() or "all agents predicted" in text.lower():
                if correct:
                    correct_unanimous += 1
            else:
                if correct:
                    correct_deviations += 1
        
        return {
            'Total Cases with Reasoning': total_cases,
            'Unanimous Agreement Cases': unanimous,
            'Deviation Cases': deviations,
            'Unanimous Percentage': unanimous / total_cases * 100 if total_cases > 0 else 0,
            'Average Reasoning Length': avg_length,
            'Accuracy in Unanimous Cases': correct_unanimous / unanimous * 100 if unanimous > 0 else 0,
            'Accuracy in Deviation Cases': correct_deviations / deviations * 100 if deviations > 0 else 0,
            'Model Mentions in Reasoning': sorted_mentions
        }

    def plot_all_models_metrics(self, output_path: str = 'Results/all_models_metrics.png'):
        """Plot performance metrics for all models including both orchestrators.
        
        Args:
            output_path (str): Path to save the plot (default: 'Results/all_models_metrics.png').
        
        Returns:
            None: Saves a bar chart comparing Accuracy, Precision, Recall, F1-Score for all models 
            and both orchestrators.
        """
        # Get metrics for all models + both orchestrators
        all_models = self.models + ['gpt-5-2025-08-07_agentic_prediction', 'gpt-5-mini-2025-08-07_agentic_prediction']
        metrics = []
        for model in all_models:
            y_true = self.df[self.true_label]
            y_pred = self.df[model]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics.append({
                'Model': model.replace('_prediction', '').replace('gpt-5-2025-08-07_agentic', 'GPT-5 Agentic').replace('gpt-5-mini-2025-08-07_agentic', 'GPT-5-Mini Agentic').replace('gpt-5-nano-2025-08-07', 'gpt-5-nano'),
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })
        df_metrics = pd.DataFrame(metrics)
        df_metrics.set_index('Model', inplace=True)
        df_metrics.plot(kind='bar', figsize=(16, 8))
        plt.title('All Models Performance Metrics Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_all_analyses_and_save(self, output_file: str = 'Results/results-mini.txt'):
        """Run all analyses and save results to file with research questions."""
        with open(output_file, 'w') as f:
            f.write("Research Questions and Results (Mini Orchestrator)\n\n")
            
            # Question 1
            f.write("1. Which LLM (AI Agent) achieves the highest accuracy in zero-shot sentiment classification?\n")
            metrics_df = self.compute_model_metrics()
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]['Model']
            f.write(f"Highest accuracy: {best_model}\n\n")
            
            # Question 2
            f.write("2. For which rating levels (1–5) do AI Agents most often fail to predict correctly?\n")
            failure_analysis = self.rating_failure_analysis()
            for model, pcts in failure_analysis.items():
                if model not in ['Overall Hardest Rating', 'Average Failure Percentages']:
                    hardest_for_model = max(pcts, key=pcts.get)
                    f.write(f"{model}: Hardest rating {hardest_for_model} ({pcts[hardest_for_model]:.2f}% failures)\n")
            f.write(f"Overall hardest rating: {failure_analysis['Overall Hardest Rating']} (avg {failure_analysis['Average Failure Percentages'][failure_analysis['Overall Hardest Rating']]:.2f}% failures)\n\n")
            
            # Question 3
            f.write("3. How does Agentic AI improve overall results through orchestration and reasoning?\n")
            value = self.system_value_analysis()
            f.write(f"Orchestrator Accuracy: {value['Orchestrator Accuracy']:.4f}, Average Model: {value['Average Model Accuracy']:.4f}, Improvement: {value['Improvement']:.4f}\n\n")
            
            # Question 4
            f.write("4. Which AI Agents contribute positively or negatively to the orchestrator’s performance?\n")
            influences_df = self.analyze_influences()
            f.write(influences_df.to_string() + "\n\n")
            
            # Question 5
            f.write("5. How frequently does the orchestrator revise its predictions based on agent input, and are such revisions beneficial or detrimental?\n")
            revisions = self.compare_orchestrator_vs_gpt5()
            f.write(f"Total Changes: {revisions['Total Changes']}, Improved: {revisions['Improved Accuracy']}, Worsened: {revisions['Worsened Accuracy']}\n\n")
            
            # Question 6
            f.write("6. How often does the orchestrator disregard agent recommendations, and is this decision advantageous?\n")
            disregard = self.orchestrator_disregard_analysis()
            f.write(f"Disregard Count: {disregard['Disregard Count']}, Percentage: {disregard['Disregard Percentage']:.2f}%, Correct: {disregard['Correct Disregards']}, Incorrect: {disregard['Incorrect Disregards']}\n\n")
            
            # Question 7
            f.write("7. Which models most strongly influence the orchestrator’s decisions, and which are least trusted?\n")
            agreement_df = self.model_agreement_with_orchestrator()
            most_influential = agreement_df.loc[agreement_df['Agreement Percentage'].idxmax()]['Model']
            least_trusted = agreement_df.loc[agreement_df['Agreement Percentage'].idxmin()]['Model']
            f.write(f"Most influential: {most_influential}, Least trusted: {least_trusted}\n\n")
            
            # Question 8
            f.write("8. How does a model’s performance as an orchestrator (in an Agentic AI setup) compare to its performance when operating independently?\n")
            comp = self.compare_orchestrator_vs_standalone()
            f.write(f"Total Changes: {comp['Total Changes']}, Improved: {comp['Improved Accuracy']}, Worsened: {comp['Worsened Accuracy']}\n\n")
            
            # Question 9
            f.write("9. Does the orchestrator primarily rely on majority voting, or does it reason directly over textual content? How often does each occur?\n")
            voting = self.voting_vs_reasoning_analysis()
            f.write(f"Matches Majority: {voting['Matches Majority']}, Differs: {voting['Differs from Majority']}, Acc when Match: {voting['Accuracy when Matches']:.4f}, Acc when Differ: {voting['Accuracy when Differs']:.4f}\n\n")
            
            # Question 10
            f.write("10. Which agents behave as outliers, potentially disrupting the overall agentic ai system?\n")
            outliers = self.outlier_agents()
            f.write(f"Outlier agents: {outliers}\n\n")
            
            # Question 11
            f.write("11. Are pre-trained LLMs without fine-tuning capable of accurately capturing sentiment from user feedback?\n")
            capability = self.pretrained_llm_capability()
            f.write(f"Average Accuracy: {capability['Average Accuracy']:.4f}, Max: {capability['Max Accuracy']:.4f}, Min: {capability['Min Accuracy']:.4f}\n\n")
            
            # Question 12
            f.write("12. Finally, is Agentic AI worth the added complexity—particularly when balancing accuracy against computational cost?\n")
            cost_benefit = self.cost_benefit_analysis()
            f.write(f"Accuracy Improvement: {cost_benefit['Accuracy Improvement']:.4f}\n")
            f.write(f"Total Individual Models Cost: ${cost_benefit['Total Individual Models Cost']:.2f}\n")
            f.write(f"Orchestrator Cost: ${cost_benefit['Orchestrator Cost']:.2f}\n")
            f.write(f"Cost Difference: ${cost_benefit['Cost Difference']:.2f}\n")
            f.write(f"Cost per Accuracy Point Improvement: ${cost_benefit['Cost per Accuracy Point Improvement']:.2f}\n")
            f.write(f"Conclusion: {cost_benefit['Conclusion']}\n\n")
            
            # Question 13
            f.write("13. Which rating is easiest for models to predict? First evaluate for each model and then the total.\n")
            easiest_analysis = self.easiest_rating_analysis()
            for model, pcts in easiest_analysis.items():
                if model not in ['Overall Easiest Rating', 'Average Success Percentages']:
                    easiest_for_model = max(pcts, key=pcts.get)
                    f.write(f"{model}: Easiest rating {easiest_for_model} ({pcts[easiest_for_model]:.2f}% success)\n")
            f.write(f"Overall easiest rating: {easiest_analysis['Overall Easiest Rating']} (avg {easiest_analysis['Average Success Percentages'][easiest_analysis['Overall Easiest Rating']]:.2f}% success)\n\n")
            
            # Question 14
            f.write("14. What insights can be gained from the orchestrator's reasoning process in the agentic AI system?\n")
            reasoning_analysis = self.analyze_orchestrator_reasoning()
            f.write(f"Total Cases with Reasoning: {reasoning_analysis['Total Cases with Reasoning']}\n")
            f.write(f"Unanimous Agreement: {reasoning_analysis['Unanimous Agreement Cases']} ({reasoning_analysis['Unanimous Percentage']:.2f}%)\n")
            f.write(f"Deviation Cases: {reasoning_analysis['Deviation Cases']}\n")
            f.write(f"Average Reasoning Length: {reasoning_analysis['Average Reasoning Length']:.0f} characters\n")
            f.write(f"Accuracy in Unanimous: {reasoning_analysis['Accuracy in Unanimous Cases']:.2f}%\n")
            f.write(f"Accuracy in Deviations: {reasoning_analysis['Accuracy in Deviation Cases']:.2f}%\n")
            f.write("Model Mentions in Reasoning (most to least):\n")
            for model, count in reasoning_analysis['Model Mentions in Reasoning']:
                f.write(f"  {model}: {count} mentions\n")
            f.write("\n")
            
            # Generate comprehensive metrics chart
            self.plot_all_models_metrics('Results/all_models_metrics.png')
            f.write("Comprehensive metrics chart saved to Results/all_models_metrics.png\n")


# Instantiate the class and run each method
if __name__ == "__main__":
    csv_path = 'Datasets/Dataset.csv'
    evaluator = AgenticEvaluation(csv_path)

    # Run all analyses and save to file
    evaluator.run_all_analyses_and_save('Results/results-gpt-5-mini-as-orchestrator.txt')
    print("All analyses completed and saved to Results/results-gpt-5-mini-as-orchestrator.txt")