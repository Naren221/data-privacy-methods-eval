import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to load and prepare the dataset
def load_data(file_path):
    """
    Load the bank marketing dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
# Exploratory data analysis
def explore_data(df):
    """
    Perform exploratory data analysis on the dataset
    """
    print("Data types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nFeature distributions:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Categorical columns: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().head())
    
    print(f"\nNumerical columns: {len(numerical_cols)}")
    
    return categorical_cols, numerical_cols

# Data preprocessing functions
def preprocess_data(df, quasi_identifiers, sensitive_attributes):
    """
    Preprocess the data for anonymization techniques
    - quasi_identifiers: Attributes that can potentially identify an individual
    - sensitive_attributes: Attributes we want to protect
    """
    df_copy = df.copy()
    
    # Encode categorical variables if needed
    encoders = {}
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            le = LabelEncoder()
            df_copy[col + '_encoded'] = le.fit_transform(df_copy[col])
            encoders[col] = le
    
    return df_copy, encoders

def generalize_age(age, age_ranges):
    """
    Generalize age into predefined ranges
    """
    for age_range, (lower, upper) in age_ranges.items():
        if lower <= age <= upper:
            return age_range
    return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"  # Fallback to decade ranges

def generalize_balance(balance, balance_ranges):
    """
    Generalize balance into predefined ranges
    """
    for balance_range, (lower, upper) in balance_ranges.items():
        if lower <= balance <= upper:
            return balance_range
    return "Other"  # Fallback

def generalize_categorical(value, mappings):
    """
    Generalize categorical attributes using predefined mappings
    """
    for general_category, specific_values in mappings.items():
        if value in specific_values:
            return general_category
    return value  # Keep original if no mapping found

# K-Anonymity Implementation
class KAnonymizer:
    def __init__(self, k=2):
        self.k = k
        self.anonymized_data = None
        self.original_data = None
        self.suppressed_records = 0
        self.generalized_columns = set()
        self.equivalence_classes = None  # Store equivalence classes for L-diversity
    
    def fit_transform(self, df, quasi_identifiers, generalization_rules=None):
        """
        Apply K-anonymity to the dataset
        """
        self.original_data = df.copy()
        df_anonymized = df.copy()
        
        # Apply generalization based on rules if provided
        if generalization_rules:
            for col, rule in generalization_rules.items():
                if col in df_anonymized.columns:
                    if col == 'age':
                        df_anonymized[col] = df_anonymized[col].apply(lambda x: generalize_age(x, rule))
                        self.generalized_columns.add(col)
                    elif col == 'balance':
                        df_anonymized[col] = df_anonymized[col].apply(lambda x: generalize_balance(x, rule))
                        self.generalized_columns.add(col)
                    elif isinstance(rule, dict):  # For categorical mappings
                        df_anonymized[col] = df_anonymized[col].apply(lambda x: generalize_categorical(x, rule))
                        self.generalized_columns.add(col)
        
        # Check if each equivalence class satisfies k-anonymity
        self.equivalence_classes = df_anonymized.groupby(quasi_identifiers)
        equivalence_class_sizes = self.equivalence_classes.size()
        
        # Identify records that violate k-anonymity
        violating_classes = equivalence_class_sizes[equivalence_class_sizes < self.k].index.tolist()
        
        # Suppress records that violate k-anonymity
        if violating_classes:
            mask = df_anonymized[quasi_identifiers].apply(tuple, axis=1).isin([tuple(x) for x in violating_classes])
            self.suppressed_records = mask.sum()
            print(f"Suppressing {self.suppressed_records} records to satisfy {self.k}-anonymity")
            df_anonymized = df_anonymized[~mask]
        
        self.anonymized_data = df_anonymized
        
        # Calculate stats
        self.total_records = len(df)
        self.remaining_records = len(df_anonymized)
        self.equivalence_class_count = len(df_anonymized.groupby(quasi_identifiers))
        
        # Calculate reidentification risk
        self.reidentification_risk = self.calculate_reidentification_risk(df_anonymized, quasi_identifiers)
        
        return df_anonymized
    
    def calculate_reidentification_risk(self, df, quasi_identifiers):
        """
        Calculate the average re-identification risk
        Risk for each record = 1 / (size of its equivalence class)
        Average risk = average of risks across all records
        """
        if df.empty:
            return 0
        
        # Group by quasi-identifiers to form equivalence classes
        eq_class_sizes = df.groupby(quasi_identifiers).size()
        
        # Map each record to its equivalence class size
        record_eq_class_sizes = df[quasi_identifiers].apply(
            lambda x: eq_class_sizes[tuple(x)], axis=1
        )
        
        # Calculate risk for each record (1/size)
        record_risks = 1 / record_eq_class_sizes
        
        # Average risk across all records
        avg_risk = record_risks.mean()
        
        return avg_risk
    
    def get_stats(self):
        """
        Return statistics about the anonymization process
        """
        if self.anonymized_data is None:
            return "No anonymization performed yet."
        
        stats = {
            "original_records": self.total_records,
            "remaining_records": self.remaining_records,
            "suppressed_records": self.suppressed_records,
            "suppression_rate": round(self.suppressed_records / self.total_records * 100, 2),
            "equivalence_class_count": self.equivalence_class_count,
            "avg_equivalence_class_size": round(self.remaining_records / self.equivalence_class_count, 2),
            "generalized_columns": list(self.generalized_columns),
            "avg_reidentification_risk": round(self.reidentification_risk, 6),
            "max_reidentification_probability": round(1/self.k, 6)  # Theoretical max based on k value
        }
        
        return stats

# L-Diversity Implementation (Applied on top of K-anonymized data)
class LDiversifier:
    def __init__(self, l=2):
        self.l = l
        self.anonymized_data = None
        self.original_data = None
        self.k_anonymized_data = None
        self.suppressed_records = 0
        self.generalized_columns = set()
    
    def fit_transform(self, k_anonymized_df, quasi_identifiers, sensitive_attributes):
        """
        Apply L-diversity to the K-anonymized dataset
        """
        self.k_anonymized_data = k_anonymized_df.copy()
        df_anonymized = k_anonymized_df.copy()
        
        # Group by quasi-identifiers
        groups = df_anonymized.groupby(quasi_identifiers)
        
        # Check L-diversity for each equivalence class
        violating_groups = []
        
        for name, group in groups:
            diverse = True
            
            # Check diversity for EACH sensitive attribute
            for sensitive_attr in sensitive_attributes:
                # Count distinct values in the sensitive attribute
                distinct_values = group[sensitive_attr].nunique()
                
                # If there are fewer than L distinct values, this group violates L-diversity
                if distinct_values < self.l:
                    diverse = False
                    break
            
            if not diverse:
                if isinstance(name, tuple):
                    violating_groups.extend([tuple(x) for x in group[quasi_identifiers].values])
                else:
                    violating_groups.extend(group[quasi_identifiers].values)
        
        # Suppress records that violate l-diversity
        if violating_groups:
            mask = df_anonymized[quasi_identifiers].apply(tuple, axis=1).isin([tuple(x) if isinstance(x, (list, np.ndarray)) else (x,) for x in violating_groups])
            self.suppressed_records = mask.sum()
            print(f"Suppressing {self.suppressed_records} records to satisfy {self.l}-diversity")
            df_anonymized = df_anonymized[~mask]
        
        self.anonymized_data = df_anonymized
        
        # Calculate stats
        self.total_records = len(k_anonymized_df)
        self.remaining_records = len(df_anonymized)
        self.equivalence_class_count = len(df_anonymized.groupby(quasi_identifiers))
        
        # Calculate reidentification risk
        self.reidentification_risk = self.calculate_reidentification_risk(df_anonymized, quasi_identifiers)
        
        return df_anonymized
    
    def calculate_reidentification_risk(self, df, quasi_identifiers):
        """
        Calculate the average re-identification risk
        Risk for each record = 1 / (size of its equivalence class)
        Average risk = average of risks across all records
        """
        if df.empty:
            return 0
        
        # Group by quasi-identifiers to form equivalence classes
        eq_class_sizes = df.groupby(quasi_identifiers).size()
        
        # Map each record to its equivalence class size
        record_eq_class_sizes = df[quasi_identifiers].apply(
            lambda x: eq_class_sizes[tuple(x)], axis=1
        )
        
        # Calculate risk for each record (1/size)
        record_risks = 1 / record_eq_class_sizes
        
        # Average risk across all records
        avg_risk = record_risks.mean()
        
        return avg_risk
    
    def get_stats(self):
        """
        Return statistics about the anonymization process
        """
        if self.anonymized_data is None:
            return "No anonymization performed yet."
        
        stats = {
            "k_anonymized_records": self.total_records,
            "remaining_records": self.remaining_records,
            "additional_suppressed_records": self.suppressed_records,
            "suppression_rate": round(self.suppressed_records / self.total_records * 100, 2),
            "equivalence_class_count": self.equivalence_class_count,
            "avg_equivalence_class_size": round(self.remaining_records / self.equivalence_class_count, 2) if self.equivalence_class_count > 0 else 0,
            "generalized_columns": list(self.generalized_columns),
            "avg_reidentification_risk": round(self.reidentification_risk, 6) if self.reidentification_risk else 0
        }
        
        return stats

# Calculate attribute level diversity
def calculate_attribute_diversity(df, quasi_identifiers, sensitive_attributes):
    """
    Calculate diversity metrics for sensitive attributes within each equivalence class
    """
    diversity_stats = {}
    
    # Group by quasi-identifiers to form equivalence classes
    groups = df.groupby(quasi_identifiers)
    
    for sensitive_attr in sensitive_attributes:
        # Track diversity metrics for this attribute
        diversity_counts = []
        
        for name, group in groups:
            # Count distinct values
            distinct_values = group[sensitive_attr].nunique()
            diversity_counts.append(distinct_values)
        
        if diversity_counts:
            diversity_stats[sensitive_attr] = {
                "min_distinct_values": min(diversity_counts),
                "max_distinct_values": max(diversity_counts),
                "avg_distinct_values": sum(diversity_counts) / len(diversity_counts),
                "median_distinct_values": np.median(diversity_counts)
            }
        else:
            diversity_stats[sensitive_attr] = {
                "min_distinct_values": 0,
                "max_distinct_values": 0,
                "avg_distinct_values": 0,
                "median_distinct_values": 0
            }
    
    return diversity_stats

# Utility Functions for Evaluation
def information_loss(original_df, anonymized_df, numerical_cols):
    """
    Calculate information loss based on numerical attributes
    """
    loss = 0
    
    # Only consider records that are present in both dataframes
    common_indices = set(original_df.index).intersection(set(anonymized_df.index))
    original_subset = original_df.loc[list(common_indices)]
    anonymized_subset = anonymized_df.loc[list(common_indices)]
    
    for col in numerical_cols:
        if col in anonymized_subset.columns and col in original_subset.columns:
            # If column was generalized, we can't directly compare
            if isinstance(anonymized_subset[col].iloc[0], str):
                loss += 1  # Maximum loss for generalized columns
            else:
                # Calculate normalized mean absolute error
                mae = np.mean(np.abs(original_subset[col] - anonymized_subset[col]))
                col_range = original_df[col].max() - original_df[col].min()
                if col_range > 0:  # Avoid division by zero
                    normalized_mae = mae / col_range
                    loss += normalized_mae
                else:
                    loss += 0
    
    # Average loss across all columns
    if len(numerical_cols) > 0:
        loss /= len(numerical_cols)
    
    return loss

def compare_techniques(original_df, k_anonymized_df, l_diversified_df, quasi_identifiers, sensitive_attributes, numerical_cols, k_stats, l_stats):
    """
    Compare K-anonymity and L-diversity approaches with risk metrics
    """
    comparison = {}
    
    # Record count metrics
    comparison['original_record_count'] = len(original_df)
    comparison['k_anon_record_count'] = len(k_anonymized_df)
    comparison['l_div_record_count'] = len(l_diversified_df)
    
    # Suppression rates
    comparison['k_anon_suppression_rate'] = k_stats['suppression_rate']
    comparison['l_div_additional_suppression_rate'] = l_stats['suppression_rate']
    comparison['l_div_total_suppression_rate'] = round(
        ((len(original_df) - len(l_diversified_df)) / len(original_df) * 100), 2
    )
    
    # Equivalence class metrics
    comparison['k_anon_eq_class_count'] = k_stats['equivalence_class_count']
    comparison['l_div_eq_class_count'] = l_stats['equivalence_class_count']
    comparison['k_anon_avg_eq_class_size'] = k_stats['avg_equivalence_class_size']
    comparison['l_div_avg_eq_class_size'] = l_stats['avg_equivalence_class_size']
    
    # Risk metrics
    comparison['k_anon_reidentification_risk'] = k_stats['avg_reidentification_risk']
    comparison['l_div_reidentification_risk'] = l_stats['avg_reidentification_risk']
    comparison['k_anon_max_reidentification_probability'] = k_stats['max_reidentification_probability']
    
    # Information loss
    comparison['k_anon_info_loss'] = information_loss(original_df, k_anonymized_df, numerical_cols)
    comparison['l_div_info_loss'] = information_loss(original_df, l_diversified_df, numerical_cols)
    
    # Diversity metrics for sensitive attributes
    k_diversity_stats = calculate_attribute_diversity(k_anonymized_df, quasi_identifiers, sensitive_attributes)
    l_diversity_stats = calculate_attribute_diversity(l_diversified_df, quasi_identifiers, sensitive_attributes)
    
    for attr in sensitive_attributes:
        for metric in ['min_distinct_values', 'avg_distinct_values']:
            comparison[f'k_anon_{attr}_{metric}'] = k_diversity_stats[attr][metric]
            comparison[f'l_div_{attr}_{metric}'] = l_diversity_stats[attr][metric]
    
    return comparison

def visualize_comparison(comparison, sensitive_attributes):
    """
    Visualize the comparison between K-anonymity and L-diversity with risk metrics
    """
    # Set up the figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot record counts
    bars1 = axes[0, 0].bar(['Original', 'K-Anonymity', 'L-Diversity'], 
                         [comparison['original_record_count'], 
                          comparison['k_anon_record_count'], 
                          comparison['l_div_record_count']])
    axes[0, 0].set_title('Record Count Comparison')
    axes[0, 0].set_ylabel('Number of Records')
    # Add data labels
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    # Plot suppression rates
    bars2 = axes[0, 1].bar(['K-Anonymity', 'L-Diversity (Additional)', 'L-Diversity (Total)'], 
                         [comparison['k_anon_suppression_rate'], 
                          comparison['l_div_additional_suppression_rate'],
                          comparison['l_div_total_suppression_rate']])
    axes[0, 1].set_title('Suppression Rate (%)')
    axes[0, 1].set_ylabel('Suppression Rate')
    # Add data labels
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom')
    
    # Plot equivalence class metrics
    x = np.arange(2)
    width = 0.35
    bars3a = axes[1, 0].bar(x - width/2, 
                          [comparison['k_anon_eq_class_count'], 
                           comparison['k_anon_avg_eq_class_size']], 
                          width, label='K-Anonymity')
    bars3b = axes[1, 0].bar(x + width/2, 
                          [comparison['l_div_eq_class_count'], 
                           comparison['l_div_avg_eq_class_size']], 
                          width, label='L-Diversity')
    axes[1, 0].set_title('Equivalence Class Metrics')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Class Count', 'Avg Class Size'])
    axes[1, 0].legend()
    # Add data labels
    for bar in bars3a:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    for bar in bars3b:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    # Plot reidentification risk
    bars4 = axes[1, 1].bar(['K-Anonymity', 'L-Diversity', 'Max Theoretical (1/k)'], 
                         [comparison['k_anon_reidentification_risk'], 
                          comparison['l_div_reidentification_risk'],
                          comparison['k_anon_max_reidentification_probability']])
    axes[1, 1].set_title('Re-identification Risk (Lower is Better)')
    axes[1, 1].set_ylabel('Risk Probability')
    # Add data labels
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom')
    
    # Plot information loss
    bars5 = axes[2, 0].bar(['K-Anonymity', 'L-Diversity'], 
                         [comparison['k_anon_info_loss'], 
                          comparison['l_div_info_loss']])
    axes[2, 0].set_title('Information Loss (Lower is Better)')
    axes[2, 0].set_ylabel('Information Loss')
    # Add data labels
    for bar in bars5:
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot diversity metrics for sensitive attributes
    x = np.arange(len(sensitive_attributes))
    width = 0.35
    metrics = ['min_distinct_values', 'avg_distinct_values']
    
    # Plot average distinct values for each sensitive attribute
    k_avg_diversity = [comparison[f'k_anon_{attr}_avg_distinct_values'] for attr in sensitive_attributes]
    l_avg_diversity = [comparison[f'l_div_{attr}_avg_distinct_values'] for attr in sensitive_attributes]
    
    bars6a = axes[2, 1].bar(x - width/2, k_avg_diversity, width, label='K-Anonymity')
    bars6b = axes[2, 1].bar(x + width/2, l_avg_diversity, width, label='L-Diversity')
    axes[2, 1].set_title('Avg Distinct Values in Sensitive Attributes')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(sensitive_attributes)
    axes[2, 1].set_ylabel('Avg Distinct Values')
    axes[2, 1].legend()
    
    # Add data labels
    for bar in bars6a:
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    for bar in bars6b:
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('privacy_comparison_with_risk.png')
    plt.close()
    
    return 'privacy_comparison_with_risk.png'

def analyze_equivalence_classes(df, quasi_identifiers, sensitive_attributes):
    """
    Analyze the equivalence classes and their diversity properties
    """
    results = {}
    
    # Group by quasi-identifiers
    equivalence_classes = df.groupby(quasi_identifiers)
    eq_class_sizes = equivalence_classes.size()
    
    # Calculate basic stats
    results['total_eq_classes'] = len(eq_class_sizes)
    results['min_eq_class_size'] = eq_class_sizes.min()
    results['max_eq_class_size'] = eq_class_sizes.max()
    results['avg_eq_class_size'] = eq_class_sizes.mean()
    
    # Analyze diversity of sensitive attributes within equivalence classes
    attribute_diversity = {}
    
    for attr in sensitive_attributes:
        diversity_counts = []
        for name, group in equivalence_classes:
            distinct_values = group[attr].nunique()
            diversity_counts.append(distinct_values)
        
        attribute_diversity[attr] = {
            'min_diversity': min(diversity_counts) if diversity_counts else 0,
            'max_diversity': max(diversity_counts) if diversity_counts else 0,
            'avg_diversity': sum(diversity_counts) / len(diversity_counts) if diversity_counts else 0,
            'classes_with_diversity_1': diversity_counts.count(1) if diversity_counts else 0,
            'classes_with_diversity_2+': sum(1 for d in diversity_counts if d >= 2) if diversity_counts else 0
        }
    
    results['attribute_diversity'] = attribute_diversity
    
    return results




# Main function
def main():
    # Load data
    file_path = "./data/bank.csv"  # Change to your file path
    df = load_data(file_path)
    
    # Explore data
    categorical_cols, numerical_cols = explore_data(df)
    
    # Define quasi-identifiers and sensitive attributes
    quasi_identifiers = ['age', 'job', 'education', 'marital']
    sensitive_attributes = ['balance', 'deposit']
    
    # Define generalization rules
    age_ranges = {
        "18-29": (18, 29),
        "30-39": (30, 39),
        "40-49": (40, 49),
        "50-59": (50, 59),
        "60+": (60, 100)
    }
    
    balance_ranges = {
        "Negative": (-10000, -1),
        "Low": (0, 999),
        "Medium": (1000, 4999),
        "High": (5000, 99999)
    }
    
    job_mappings = {
        "Office Worker": ["admin.", "management", "technician"],
        "Manual Labor": ["blue-collar", "services"],
        "Professional": ["entrepreneur", "self-employed"],
        "Other": ["unemployed", "housemaid", "student", "retired", "unknown"]
    }
    
    education_mappings = {
        "Basic": ["primary", "unknown"],
        "Intermediate": ["secondary"],
        "Advanced": ["tertiary"]
    }
    
    generalization_rules = {
        "age": age_ranges,
        "balance": balance_ranges,
        "job": job_mappings,
        "education": education_mappings
    }
    
    # Preprocess data
    processed_df, encoders = preprocess_data(df, quasi_identifiers, sensitive_attributes)
    
    # Apply K-anonymity
    k = 5 # Set k value for k-anonymity
    k_anonymizer = KAnonymizer(k=k)
    k_anonymized_df = k_anonymizer.fit_transform(processed_df, quasi_identifiers, generalization_rules)
    k_stats = k_anonymizer.get_stats()
    print(f"\nK-Anonymity Statistics (k={k}):")
    for key, value in k_stats.items():
        print(f"{key}: {value}")
    print("K Statistics ", k_stats)
    # Apply L-diversity
    l = 2  # Set l value for l-diversity
    l_diversifier = LDiversifier(l=l)
    l_diversified_df = l_diversifier.fit_transform(k_anonymized_df, quasi_identifiers, sensitive_attributes)
    l_stats = l_diversifier.get_stats()
    print(f"\nL-Diversity Statistics (l={l}):")
    for key, value in l_stats.items():
        print(f"{key}: {value}")
    
    # Compare techniques
    comparison = compare_techniques(processed_df, k_anonymized_df, l_diversified_df,quasi_identifiers, sensitive_attributes, numerical_cols, k_stats, l_stats)
    print("\nComparison of K-Anonymity and L-Diversity:")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    
    # Visualize comparison
    visualization_path = visualize_comparison(comparison, sensitive_attributes)
    print(f"\nComparison visualization saved to {visualization_path}")
    
    # Save anonymized datasets if needed
    k_anonymized_df.to_csv("k_anonymized_data.csv", index=False)
    l_diversified_df.to_csv("l_diversified_data.csv", index=False)
    
    return k_anonymized_df, l_diversified_df, comparison

if __name__ == "__main__":
    main()