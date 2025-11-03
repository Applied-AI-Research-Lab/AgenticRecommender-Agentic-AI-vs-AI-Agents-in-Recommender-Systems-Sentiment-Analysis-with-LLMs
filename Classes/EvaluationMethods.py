import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns

class EvaluationMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        # Get the directory where this script is located, then go up one level to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self.pre_path = os.path.join(project_root, 'Datasets') + os.sep

    def evaluate_results(self, original, prediction, model_name):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        accuracy = round(accuracy_score(data[original], data[prediction]), 4)
        precision = round(precision_score(data[original], data[prediction], average='weighted'), 4)
        recall = round(recall_score(data[original], data[prediction], average='weighted'), 4)
        f1 = round(f1_score(data[original], data[prediction], average='weighted'), 4)

        # Create a DataFrame with the evaluation results including the 'model' column
        evaluation_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1]
        })

        # Append the results to the existing CSV file or create a new one
        evaluation_df.to_csv(self.pre_path + 'evaluation-results.csv', mode='a',
                             header=not os.path.exists(self.pre_path + 'evaluation-results.csv'), index=False)

        return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    def scatterplot(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)
        prediction = df[prediction_column]
        original = df[original_column]

        # Calculate Mean Absolute Error
        mae = abs(original - prediction).mean()

        # Create a scatter plot with a regression line
        sns.regplot(x=original, y=prediction, scatter_kws={'alpha': 0.5})

        plt.xlabel(original_column)
        plt.ylabel(prediction_column)

        # Save the scatterplot image to the Datasets folder
        plt.savefig(os.path.join(self.pre_path + 'Plots/', prediction_column + '.png'))

        # Show the plot
        plt.show()

        return mae

    def count_matching_rows(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Count the number of same value rows
        matching_rows = df[df[original_column] == df[prediction_column]]

        return prediction_column, len(matching_rows)

    def plot_histograms(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Separate predicted probabilities by class
        predicted_probabilities_class_0 = dataframe.loc[dataframe[original_column] == 0, prediction_column]
        predicted_probabilities_class_1 = dataframe.loc[dataframe[original_column] == 1, prediction_column]

        # Plot histograms
        plt.figure(figsize=(10, 5))

        # Histogram for class 0
        plt.subplot(1, 2, 1)
        plt.hist(predicted_probabilities_class_0, bins=20, color='blue', alpha=0.7)
        plt.title('Predicted Probabilities - Class 0')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        # Histogram for class 1
        plt.subplot(1, 2, 2)
        plt.hist(predicted_probabilities_class_1, bins=20, color='orange', alpha=0.7)
        plt.title('Predicted Probabilities - Class 1')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Extract data from DataFrame
        y_true = dataframe[original_column]
        y_pred = dataframe[prediction_column]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix \n('+prediction_column+')')
        plt.show()

    """
    Plot a stacked bar chart showing the distribution of labels across categories in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """

    def plot_stacked_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])
        # Calculate row-wise percentages
        cross_tab_percent = cross_tab.apply(lambda x: x * 100 / x.sum(), axis=1)

        # Plotting the stacked bar chart
        ax = cross_tab_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Adding labels and title
        plt.title(f'Stacked Bar Chart of {original_column} vs. {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)

        # Adding percentages as text on each bar segment
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8)

        plt.show()

    """
    Plot a grouped bar chart showing the relationship between labels in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """
    def plot_grouped_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        pivot_table = data.groupby([original_column, prediction_column]).size().unstack(fill_value=0)
        pivot_table.plot(kind='bar', figsize=(10, 6))
        plt.title(f'Relationship between {original_column} and {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    """
    Plot a heatmap showing relationships and patterns between label categories in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """

    def plot_heatmap(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])

        plt.figure(figsize=(12, 10))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')

        plt.title(f'Heatmap of {original_column} vs. {prediction_column}')
        plt.xlabel(prediction_column)
        plt.ylabel(original_column)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Get project root and create Plots directory there
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "Plots")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'heatmap_{original_column}_vs_{prediction_column}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.show()
        print(f"Heatmap saved at: {output_path}")


# Example Usage
# Instantiate the DatasetMethods class by providing the (dataset_path)
EVM = EvaluationMethods(dataset_path='Dataset.csv')
# rm -rf .venv
# python3 -m venv .venv
# source .venv/bin/activate && pip install pandas matplotlib scikit-learn seaborn anthropic openai python-dotenv
# /Users/rkonstadinos/PycharmProjects/AgenticRecommender/.venv/bin/python Classes/EvaluationMethods.py

# # Count correct predictions
# print(str(EVM.count_matching_rows('rating', 'gemini-2.5-pro_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gemini-2.5-flash_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gemini-2.5-flash-lite_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-5_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-5-mini_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-5-nano-2025-08-07_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-4.1_prediction')))
# print(str(EVM.count_matching_rows('rating', 'claude-sonnet-4-5_prediction')))
# print(str(EVM.count_matching_rows('rating', 'claude-haiku-4-5_prediction')))
# print(str(EVM.count_matching_rows('rating', 'claude-opus-4-1_prediction')))
# print(str(EVM.count_matching_rows('rating', 'deepseek-chat_prediction')))
# print(str(EVM.count_matching_rows('rating', 'deepseek-reasoner_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-5-2025-08-07_agentic_prediction')))
# print(str(EVM.count_matching_rows('rating', 'gpt-5-mini-2025-08-07_agentic_prediction')))

# # Evaluate the predictions made by each model
# print(f'gemini-2.5-pro: ' + str(EVM.evaluate_results('rating', 'gemini-2.5-pro_prediction', 'gemini-2.5-pro')))
# print(f'gemini-2.5-flash: ' + str(EVM.evaluate_results('rating', 'gemini-2.5-flash_prediction', 'gemini-2.5-flash')))
# print(f'gemini-2.5-flash-lite: ' + str(EVM.evaluate_results('rating', 'gemini-2.5-flash-lite_prediction', 'gemini-2.5-flash-lite')))
# print(f'gpt-5: ' + str(EVM.evaluate_results('rating', 'gpt-5_prediction', 'gpt-5')))
# print(f'gpt-5-mini: ' + str(EVM.evaluate_results('rating', 'gpt-5-mini_prediction', 'gpt-5-mini')))
# print(f'gpt-5-nano: ' + str(EVM.evaluate_results('rating', 'gpt-5-nano-2025-08-07_prediction', 'gpt-5-nano')))
# print(f'gpt-4.1: ' + str(EVM.evaluate_results('rating', 'gpt-4.1_prediction', 'gpt-4.1')))
# print(f'claude-sonnet-4-5: ' + str(EVM.evaluate_results('rating', 'claude-sonnet-4-5_prediction', 'claude-sonnet-4-5')))
# print(f'claude-haiku-4-5: ' + str(EVM.evaluate_results('rating', 'claude-haiku-4-5_prediction', 'claude-haiku-4-5')))
# print(f'claude-opus-4-1: ' + str(EVM.evaluate_results('rating', 'claude-opus-4-1_prediction', 'claude-opus-4-1')))
# print(f'deepseek-chat: ' + str(EVM.evaluate_results('rating', 'deepseek-chat_prediction', 'deepseek-chat')))
# print(f'deepseek-reasoner: ' + str(EVM.evaluate_results('rating', 'deepseek-reasoner_prediction', 'deepseek-reasoner')))
# print(f'GPT-5 as an Orchestrator: ' + str(EVM.evaluate_results('rating', 'gpt-5-2025-08-07_agentic_prediction', 'GPT-5 as an Orchestrator')))
# print(f'GPT-5-mini as an Orchestrator: ' + str(EVM.evaluate_results('rating', 'gpt-5-mini-2025-08-07_agentic_prediction', 'GPT-5-mini as an Orchestrator')))

# Plots
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gemini-2.5-pro_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gemini-2.5-flash_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gemini-2.5-flash-lite_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-5_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-5-mini_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-5-nano-2025-08-07_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-4.1_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='claude-sonnet-4-5_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='claude-haiku-4-5_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='claude-opus-4-1_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='deepseek-chat_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='deepseek-reasoner_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-5-2025-08-07_agentic_prediction'))
# print(EVM.plot_heatmap(original_column='rating', prediction_column='gpt-5-mini-2025-08-07_agentic_prediction'))