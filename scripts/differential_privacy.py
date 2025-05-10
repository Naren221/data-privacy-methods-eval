import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0):
        """
        Initialize Differential Privacy with a privacy budget epsilon.
        
        Parameters:
        -----------
        epsilon : float
            Privacy budget. Lower values provide stronger privacy guarantees
            but reduce accuracy. Typically between 0.1 and 10.
        """
        self.epsilon = epsilon
        self.anonymized_data = None
        self.original_data = None
        self.sensitivity = {}
        self.encoders = {}
        self.noise_added = {}
        self.info_loss = {}
        
    def fit_transform(self, df, quasi_identifiers, sensitive_attributes, numerical_cols=None):
        """
        Apply differential privacy to the sensitive attributes.
        """
        self.original_data = df.copy()
        df_anonymized = df.copy()
        self.total_records = len(df)  # ✅ Moved up before _add_laplace_noise is called

        for col in sensitive_attributes:
            if df[col].dtype in [np.int64, np.float64]:
                df_anonymized, noise_info = self._add_laplace_noise(df_anonymized, col)
                self.noise_added[col] = noise_info
            else:
                df_anonymized = self._apply_exponential_mechanism(df_anonymized, col)

        self.anonymized_data = df_anonymized
        self.info_loss = self._calculate_info_loss(numerical_cols)

        return df_anonymized

    
    def _add_laplace_noise(self, df, column):
        """
        Add Laplace noise to a numerical column.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Dataset to modify
        column : str
            Column name to add noise to
            
        Returns:
        --------
        pandas DataFrame, dict
            Modified dataset and noise statistics
        """
        # Ensure the column is numeric
        if df[column].dtype == object:
            print(f"Warning: Column {column} is not numeric. Skipping Laplace noise.")
            return df, {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'scale': 0}
            
        # Convert to numeric values if needed
        data = pd.to_numeric(df[column], errors='coerce').values
        
        # Calculate sensitivity (use range for this implementation)
        sensitivity = (np.nanmax(data) - np.nanmin(data)) / self.total_records
        self.sensitivity[column] = sensitivity
        
        # Scale parameter for Laplace noise (ensure it's positive)
        scale = max(sensitivity / self.epsilon, 1e-10)
        
        # Generate and add Laplace noise
        noise = np.random.laplace(0, scale, size=len(data))
        df[column] = data + noise
        
        # Record noise statistics
        noise_info = {
            'mean': float(np.mean(noise)),
            'std': float(np.std(noise)),
            'min': float(np.min(noise)),
            'max': float(np.max(noise)),
            'scale': float(scale)
        }
        
        return df, noise_info
    
    def _apply_exponential_mechanism(self, df, column):
        """
        Apply the exponential mechanism to a categorical column.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Dataset to modify
        column : str
            Column name to apply the mechanism to
            
        Returns:
        --------
        pandas DataFrame
            Modified dataset
        """
        # Check if column exists and is categorical
        if column not in df.columns:
            print(f"Warning: Column {column} not found in dataframe. Skipping.")
            return df
            
        # For categorical data, use category frequencies to create a utility function
        value_counts = df[column].value_counts()
        
        if len(value_counts) == 0:
            print(f"Warning: Column {column} has no values. Skipping.")
            return df
            
        # Apply exponential mechanism - more frequent values have higher probability
        # of being selected as replacements
        values = list(value_counts.index)
        freq = list(value_counts.values)
        
        # Utility scores (using frequencies as proxies for utility)
        utility_scores = np.array(freq) / np.sum(freq)
        
        # Apply exponential mechanism by sampling from utility distribution
        # Calculate probabilities with differential privacy once for efficiency
        probs = np.exp(self.epsilon * utility_scores / 2)
        probs = probs / np.sum(probs)
        
        # Create a copy of the column to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Sample new values for the entire column at once
        new_values = np.random.choice(values, size=len(df), p=probs)
        df_copy[column] = new_values
        
        return df_copy
    
    def _calculate_info_loss(self, numerical_cols):
        """
        Calculate information loss based on numerical attributes.
        
        Parameters:
        -----------
        numerical_cols : list
            List of numerical columns to consider
            
        Returns:
        --------
        dict
            Dictionary with information loss metrics
        """
        info_loss = {}
        
        if numerical_cols is None or self.anonymized_data is None or self.original_data is None:
            return {'overall': 0}
        
        loss_values = []
        
        for col in numerical_cols:
            if col in self.anonymized_data.columns and col in self.original_data.columns:
                # Calculate normalized mean absolute error
                mae = np.mean(np.abs(self.original_data[col] - self.anonymized_data[col]))
                col_range = self.original_data[col].max() - self.original_data[col].min()
                
                if col_range > 0:  # Avoid division by zero
                    normalized_mae = mae / col_range
                    info_loss[col] = normalized_mae
                    loss_values.append(normalized_mae)
                else:
                    info_loss[col] = 0
                    loss_values.append(0)
        
        # Calculate overall information loss
        if loss_values:
            info_loss['overall'] = sum(loss_values) / len(loss_values)
        else:
            info_loss['overall'] = 0
            
        return info_loss
    
    def get_stats(self):
        """
        Return statistics about the differential privacy process.
        
        Returns:
        --------
        dict
            Dictionary with various statistics
        """
        if self.anonymized_data is None:
            return "No anonymization performed yet."
        
        stats = {
            "epsilon": self.epsilon,
            "original_records": self.total_records,
            "remaining_records": len(self.anonymized_data),  # Always same as original
            "suppressed_records": 0,  # DP doesn't suppress records
            "suppression_rate": 0,  # DP doesn't suppress records
            "avg_info_loss": self.info_loss.get('overall', 0),
            "noise_added": self.noise_added,
            "reidentification_risk": self._calculate_reidentification_risk()
        }
        
        return stats
    
    def _calculate_reidentification_risk(self):
        """
        Estimate reidentification risk based on the privacy budget.
        
        For differential privacy, risk is related to the privacy parameter epsilon.
        Higher epsilon means higher risk.
        
        Returns:
        --------
        float
            Estimated reidentification risk
        """
        # For differential privacy, theoretical privacy guarantee is e^epsilon
        # We normalize this to a probability between 0 and 1
        privacy_guarantee = np.exp(self.epsilon)
        risk = min(1.0, privacy_guarantee / (1 + privacy_guarantee))
        return risk


def compare_with_other_techniques(dp_stats, k_stats, l_stats, numerical_cols):
    """
    Compare Differential Privacy with K-Anonymity and L-Diversity.
    
    Parameters:
    -----------
    dp_stats : dict
        Statistics from Differential Privacy
    k_stats : dict
        Statistics from K-Anonymity
    l_stats : dict
        Statistics from L-Diversity
    numerical_cols : list
        List of numerical columns used for info loss calculation
    
    Returns:
    --------
    dict
        Comparison of the three techniques
    """
    comparison = {}
    
    # Record count metrics
    comparison['dp_record_count'] = dp_stats['remaining_records']
    comparison['k_anon_record_count'] = k_stats['remaining_records']
    comparison['l_div_record_count'] = l_stats['remaining_records']
    
    # Suppression rates
    comparison['dp_suppression_rate'] = dp_stats['suppression_rate']
    comparison['k_anon_suppression_rate'] = k_stats['suppression_rate']
    comparison['l_div_suppression_rate'] = l_stats['suppression_rate']
    
    # Information loss
    comparison['dp_info_loss'] = dp_stats['avg_info_loss']
    comparison['k_anon_info_loss'] = k_stats.get('avg_info_loss', 0)
    comparison['l_div_info_loss'] = l_stats.get('avg_info_loss', 0)
    
    # Risk metrics
    comparison['dp_reidentification_risk'] = dp_stats['reidentification_risk']
    comparison['k_anon_reidentification_risk'] = k_stats['avg_reidentification_risk']
    comparison['l_div_reidentification_risk'] = l_stats['avg_reidentification_risk']
    
    return comparison


def load_data(file_path):
    """
    Load the bank marketing dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def analyze_privacy_techniques(original_df, epsilon_values=None):
    """
    Analyze differential privacy with various epsilon values.
    
    Parameters:
    -----------
    original_df : pandas DataFrame
        Original dataset
    epsilon_values : list, optional
        List of epsilon values to test
        
    Returns:
    --------
    dict
        Dictionary with results for each epsilon value
    """
    if epsilon_values is None:
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = {}
    
    # Define data attributes
    quasi_identifiers = ['age', 'job', 'education', 'marital']
    sensitive_attributes = ['balance', 'deposit']
    numerical_cols = ['age', 'balance']
    
    for epsilon in epsilon_values:
        print(f"\nApplying Differential Privacy with epsilon={epsilon}")
        dp = DifferentialPrivacy(epsilon=epsilon)
        dp_anonymized = dp.fit_transform(original_df, quasi_identifiers, sensitive_attributes, numerical_cols)
        dp_stats = dp.get_stats()
        
        print(f"Differential Privacy Statistics (epsilon={epsilon}):")
        for key, value in dp_stats.items():
            if key != 'noise_added':  # Skip detailed noise info for readability
                print(f"{key}: {value}")
        
        results[epsilon] = {
            'stats': dp_stats,
            'anonymized_data': dp_anonymized
        }
    
    return results


def main():
    # Load data
    try:
        file_path = "./data/bank.csv"  # Change to your file path
        df = load_data(file_path)
        
        if df is None:
            return
        
        # Define data attributes
        quasi_identifiers = ['age', 'job', 'education', 'marital']
        sensitive_attributes = ['balance', 'deposit']
        numerical_cols = ['age', 'balance']
        
        # Check if required columns exist
        missing_columns = [col for col in quasi_identifiers + sensitive_attributes if col not in df.columns]
        if missing_columns:
            print(f"Error: The following required columns are missing from the dataset: {missing_columns}")
            return
        
        # Apply differential privacy with a specific epsilon
        epsilon = 0.01  # A moderate privacy budget
        print(f"\nApplying Differential Privacy with epsilon={epsilon}")
        dp = DifferentialPrivacy(epsilon=epsilon)
        dp_anonymized = dp.fit_transform(df, quasi_identifiers, sensitive_attributes, numerical_cols)
        dp_stats = dp.get_stats()
        
        print(f"Differential Privacy Statistics (epsilon={epsilon}):")
        for key, value in dp_stats.items():
            if key != 'noise_added':  # Skip detailed noise info for readability
                print(f"{key}: {value}")
        
        # Save the anonymized dataset
        dp_anonymized.to_csv("./output/dp_anonymized_data.csv", index=False)
        print(f"Anonymized data saved to dp_anonymized_data.csv")
        
        print("\nTry different epsilon values to explore privacy-utility tradeoffs:")
        print("Lower epsilon (e.g., 0.1): Stronger privacy but lower utility")
        print("Higher epsilon (e.g., 5.0): Lower privacy but higher utility")
        
        # Uncomment to analyze with different epsilon values
        # epsilon_results = analyze_privacy_techniques(df, [0.1, 1.0, 10.0])
        
        return dp_anonymized, dp_stats
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging


if __name__ == "__main__":
    main()