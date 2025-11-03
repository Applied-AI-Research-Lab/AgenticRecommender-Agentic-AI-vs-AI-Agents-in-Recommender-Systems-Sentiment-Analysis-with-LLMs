"""
DatasetMethods.py
This module handles dataset sampling and combination from multiple balanced CSV files.
It creates a combined dataset by randomly sampling equal numbers from each rating label.
"""

import pandas as pd
import os


class DatasetMethods:
    def __init__(self, datasets_path='Datasets/'):
        """
        Initialize the DatasetMethods class.
        
        Args:
            datasets_path (str): Path to the datasets directory. Default is 'Datasets/'
        """
        self.datasets_path = datasets_path
        
        # Define the source dataset files
        self.source_datasets = [
            'Amazon_Fashion_balanced.csv',
            'Automotive_balanced.csv',
            'Books_balanced.csv',
            'Electronics_balanced.csv',
            'Video_Games_balanced.csv'
        ]
        
        # Output combined dataset filename
        self.output_dataset = 'Dataset.csv'
        
    def get_balanced_sample(self, df, samples_per_label=200, label_column='rating'):
        """
        Get a balanced sample from a dataframe by sampling equally from each label.
        
        Args:
            df (pd.DataFrame): The input dataframe
            samples_per_label (int): Number of samples to get per label. Default is 200
            label_column (str): The column name containing labels. Default is 'rating'
            
        Returns:
            pd.DataFrame: A balanced sample with equal samples from each label
        """
        sampled_dfs = []
        
        # Get unique labels (ratings)
        unique_labels = sorted(df[label_column].unique())
        
        print(f"  Found labels: {unique_labels}")
        
        # Sample from each label
        for label in unique_labels:
            label_df = df[df[label_column] == label]
            
            # Check if we have enough samples for this label
            available_samples = len(label_df)
            samples_to_take = min(samples_per_label, available_samples)
            
            if available_samples < samples_per_label:
                print(f"  Warning: Label {label} only has {available_samples} samples, "
                      f"taking all available instead of {samples_per_label}")
            
            # Random sample from this label
            sampled = label_df.sample(n=samples_to_take, random_state=42)
            sampled_dfs.append(sampled)
            
            print(f"  Sampled {samples_to_take} rows for label {label}")
        
        # Combine all sampled dataframes
        balanced_sample = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the combined dataset
        balanced_sample = balanced_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_sample
    
    def combine_datasets(self, samples_per_label=200):
        """
        Combine multiple balanced datasets by sampling from each.
        
        Args:
            samples_per_label (int): Number of samples per label per dataset. Default is 200
            
        Returns:
            dict: Status and information about the combined dataset
        """
        all_samples = []
        
        print("=" * 70)
        print("Starting Dataset Combination Process")
        print("=" * 70)
        print(f"Samples per label per dataset: {samples_per_label}")
        print(f"Expected total samples: {len(self.source_datasets) * samples_per_label * 5} rows")
        print("=" * 70)
        
        # Process each dataset
        for idx, dataset_file in enumerate(self.source_datasets, 1):
            dataset_path = os.path.join(self.datasets_path, dataset_file)
            
            print(f"\n[{idx}/{len(self.source_datasets)}] Processing: {dataset_file}")
            
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"  ERROR: File not found at {dataset_path}")
                continue
            
            try:
                # Read the dataset
                df = pd.read_csv(dataset_path)
                print(f"  Loaded {len(df)} rows")
                
                # Get balanced sample
                sampled_df = self.get_balanced_sample(df, samples_per_label=samples_per_label)
                
                # Add a source column to track which dataset this came from
                sampled_df['source_dataset'] = dataset_file.replace('_balanced.csv', '')
                
                all_samples.append(sampled_df)
                print(f"  Added {len(sampled_df)} rows to combined dataset")
                
            except Exception as e:
                print(f"  ERROR processing {dataset_file}: {str(e)}")
                continue
        
        # Combine all samples
        if not all_samples:
            return {
                "status": False,
                "message": "No datasets were successfully processed"
            }
        
        print("\n" + "=" * 70)
        print("Combining all samples...")
        combined_df = pd.concat(all_samples, ignore_index=True)
        
        # Shuffle the final combined dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        output_path = os.path.join(self.datasets_path, self.output_dataset)
        combined_df.to_csv(output_path, index=False)
        
        print("=" * 70)
        print("Dataset Combination Complete!")
        print("=" * 70)
        print(f"Total rows in combined dataset: {len(combined_df)}")
        print(f"Output file: {output_path}")
        
        # Show distribution by rating
        print("\nRating distribution:")
        rating_counts = combined_df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  Rating {rating}: {count} rows")
        
        # Show distribution by source dataset
        print("\nSource dataset distribution:")
        source_counts = combined_df['source_dataset'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} rows")
        
        print("=" * 70)
        
        return {
            "status": True,
            "message": f"Successfully combined {len(self.source_datasets)} datasets",
            "total_rows": len(combined_df),
            "output_file": output_path,
            "rating_distribution": rating_counts.to_dict(),
            "source_distribution": source_counts.to_dict()
        }
    
    def verify_combined_dataset(self):
        """
        Verify the combined dataset and show statistics.
        
        Returns:
            dict: Statistics about the combined dataset
        """
        output_path = os.path.join(self.datasets_path, self.output_dataset)
        
        if not os.path.exists(output_path):
            return {
                "status": False,
                "message": f"Combined dataset not found at {output_path}"
            }
        
        df = pd.read_csv(output_path)
        
        print("=" * 70)
        print("Combined Dataset Verification")
        print("=" * 70)
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        print("\nRating distribution:")
        rating_counts = df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  Rating {rating}: {count} rows ({count/len(df)*100:.1f}%)")
        
        if 'source_dataset' in df.columns:
            print("\nSource dataset distribution:")
            source_counts = df['source_dataset'].value_counts()
            for source, count in source_counts.items():
                print(f"  {source}: {count} rows ({count/len(df)*100:.1f}%)")
        
        print("=" * 70)
        
        return {
            "status": True,
            "total_rows": len(df),
            "rating_distribution": rating_counts.to_dict(),
            "source_distribution": source_counts.to_dict() if 'source_dataset' in df.columns else {}
        }


# Main execution
if __name__ == "__main__":
    # Create an instance of DatasetMethods
    # Use relative path from the Classes directory
    dataset_methods = DatasetMethods(datasets_path='../Datasets/')
    
    # Combine the datasets with 200 samples per label (5 labels) = 1000 samples per dataset
    # 5 datasets × 1000 samples = 5000 total samples
    result = dataset_methods.combine_datasets(samples_per_label=200)
    
    if result['status']:
        print("Dataset combination successful!")
        
        # Verify the combined dataset
        print("\n")
        verification = dataset_methods.verify_combined_dataset()
    else:
        print(f"Error: {result['message']}")
