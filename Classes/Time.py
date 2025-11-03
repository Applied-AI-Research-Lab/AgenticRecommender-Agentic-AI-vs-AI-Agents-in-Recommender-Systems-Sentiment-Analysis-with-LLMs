import pandas as pd

print('Model,Mean Prediction Time,Total Time (for 5,000 predictions)')
def sum_and_average_column(dataset: str, column: str) -> tuple:
    df = pd.read_csv(dataset)
    total = round(df[column].sum(), 2)
    average = round(df[column].mean(), 2)
    return column + ',' + str(average) + ',' + str(total)
    # return column, total, average


# print(sum_and_average_column("Datasets/Dataset.csv", "time-gemini-2.5-pro"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gemini-2.5-flash"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gemini-2.5-flash-lite"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gpt-5-2025-08-07"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gpt-5-mini-2025-08-07"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gpt-5-nano-2025-08-07"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-gpt-4.1-2025-04-14"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-claude-sonnet-4-5"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-claude-haiku-4-5"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-claude-opus-4-1"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-deepseek-chat"))
# print(sum_and_average_column("Datasets/Dataset.csv", "time-deepseek-reasoner"))

# Results:
# Model,Mean Prediction Time,Total Time (for 5,000 predictions)
# time-gemini-2.5-pro,9.83,49165.83
# time-gemini-2.5-flash,5.33,26665.66
# time-gemini-2.5-flash-lite,0.77,3860.23
# time-gpt-5-2025-08-07,9.41,47045.13
# time-gpt-5-mini-2025-08-07,12.39,61965.27
# time-gpt-5-nano-2025-08-07,4.08,20377.21
# time-gpt-4.1-2025-04-14,1.42,7101.79
# time-claude-sonnet-4-5,2.66,13275.2
# time-claude-haiku-4-5,1.41,7025.51
# time-claude-opus-4-1,3.85,19251.89
# time-deepseek-chat,1.68,8386.15
# time-deepseek-reasoner,31.31,156556.33