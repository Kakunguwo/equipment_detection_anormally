
"""
Equipment Anomaly Detection using Machine Learning

This script implements Random Forest and SVM classifiers to detect equipment
anomalies from sensor data. Designed for biomedical equipment monitoring.

Usage:
    python main.py --model all          # Run both models (default)
    python main.py --model rf           # Run Random Forest only
    python main.py --model svm          # Run SVM only
    python main.py --data custom.csv    # Use custom dataset

Authors: Ronnie Kakunguwo, Munashe Dube, Theophelous Manyere, Ratidzo Munyikwa, Brandon Mubairatsunga, Tariro Chidziva, Fransinca Gombani, Chido Mujuru, Nicola Kitumetsi Chiware
Course: Conditional Monitoring of Intelligent Machines
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Scikit-learn imports
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def load_and_explore_data(data_path):
    """
    Load the equipment sensor data and perform initial exploration.
    
    Args:
        data_path (str): Path to the CSV data file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("=" * 60)
    print("LOADING AND EXPLORING DATA")
    print("=" * 60)
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully from: {data_path}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Display basic statistics
        print("\n--- Dataset Statistics ---")
        print(df.describe())
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print("\n⚠️  Warning: Missing values detected!")
            print(df.isnull().sum())
        else:
            print("\n✓ No missing values found")
            
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_path}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        sys.exit(1)


def preprocess_data(df):
    """
    Clean and preprocess the data for machine learning.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X_scaled, y, scaler, feature_names) - preprocessed data
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Remove metadata columns (equipment ID, location)
    columns_to_drop = ["equipment", "location"]
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols:
        df_cleaned = df.drop(columns=existing_cols)
        print(f"✓ Removed metadata columns: {existing_cols}")
    else:
        df_cleaned = df.copy()
        print("ℹ️  No metadata columns to remove")
    
    # Separate features and target
    if "faulty" not in df_cleaned.columns:
        print("❌ Error: 'faulty' column not found in dataset!")
        sys.exit(1)
    
    X = df_cleaned.drop(columns=["faulty"])
    y = df_cleaned["faulty"]
    
    print(f"✓ Features (X): {X.shape} - {list(X.columns)}")
    print(f"✓ Target (y): {y.shape} - Classes: {y.value_counts().to_dict()}")
    
    # Scale features for SVM (Random Forest doesn't require scaling but won't hurt)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Features standardized using StandardScaler")
    
    return X_scaled, y, scaler, list(X.columns)


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.array): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"✓ Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"✓ Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (np.array): Training features
        y_train (pd.Series): Training targets
        n_estimators (int): Number of trees
        random_state (int): Random seed
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    print(f"✓ Random Forest trained with {n_estimators} trees")
    return rf


def train_svm(X_train, y_train, C=1.0, gamma='scale', random_state=42):
    """
    Train an SVM classifier with RBF kernel.
    
    Args:
        X_train (np.array): Training features
        y_train (pd.Series): Training targets
        C (float): Regularization parameter
        gamma (str/float): Kernel coefficient
        random_state (int): Random seed
        
    Returns:
        SVC: Trained model
    """
    print("\n--- Training SVM ---")
    svm = SVC(C=C, gamma=gamma, random_state=random_state)
    svm.fit(X_train, y_train)
    print(f"✓ SVM trained with C={C}, gamma={gamma}")
    return svm


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained classifier
        X_test (np.array): Test features
        y_test (pd.Series): Test targets
        model_name (str): Name of the model for display
        
    Returns:
        tuple: (predictions, classification_report_dict)
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    print(f"✓ {model_name} Accuracy: {accuracy:.3f}")
    print(f"✓ {model_name} Precision (weighted): {report['weighted avg']['precision']:.3f}")
    print(f"✓ {model_name} Recall (weighted): {report['weighted avg']['recall']:.3f}")
    print(f"✓ {model_name} F1-Score (weighted): {report['weighted avg']['f1-score']:.3f}")
    
    return predictions, report


def create_comparison_table(rf_report, svm_report, rf_accuracy, svm_accuracy):
    """
    Create a formatted comparison table of model performance.
    
    Args:
        rf_report (dict): Random Forest classification report
        svm_report (dict): SVM classification report
        rf_accuracy (float): Random Forest accuracy
        svm_accuracy (float): SVM accuracy
    """
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print(f"{'Metric':<15} {'Random Forest':<15} {'SVM':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<15} {rf_accuracy:<15.3f} {svm_accuracy:<15.3f}")
    print(f"{'Precision':<15} {rf_report['weighted avg']['precision']:<15.3f} {svm_report['weighted avg']['precision']:<15.3f}")
    print(f"{'Recall':<15} {rf_report['weighted avg']['recall']:<15.3f} {svm_report['weighted avg']['recall']:<15.3f}")
    print(f"{'F1-Score':<15} {rf_report['weighted avg']['f1-score']:<15.3f} {svm_report['weighted avg']['f1-score']:<15.3f}")


def plot_metrics_comparison(rf_report, svm_report, save_path="images/metrics_comparison.png"):
    """
    Create a bar chart comparing model metrics.
    
    Args:
        rf_report (dict): Random Forest classification report
        svm_report (dict): SVM classification report
        save_path (str): Path to save the plot
    """
    print("\n--- Creating Metrics Comparison Plot ---")
    
    # Extract weighted average metrics
    metrics = ['precision', 'recall', 'f1-score']
    rf_metrics = [rf_report['weighted avg'][m] for m in metrics]
    svm_metrics = [svm_report['weighted avg'][m] for m in metrics]
    
    # Create plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, svm_metrics, width, label='SVM', 
                   color='#ff7f0e', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Weighted Average Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.show()


def plot_confusion_matrices(y_test, rf_predictions, svm_predictions, 
                          save_path="images/confusion_matrices.png"):
    """
    Create side-by-side confusion matrices for both models.
    
    Args:
        y_test (pd.Series): True labels
        rf_predictions (np.array): Random Forest predictions
        svm_predictions (np.array): SVM predictions
        save_path (str): Path to save the plot
    """
    print("\n--- Creating Confusion Matrices ---")
    
    # Generate confusion matrices
    rf_cm = confusion_matrix(y_test, rf_predictions)
    svm_cm = confusion_matrix(y_test, svm_predictions)
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Random Forest Confusion Matrix
    ConfusionMatrixDisplay(rf_cm, display_labels=['Normal', 'Faulty']).plot(
        ax=ax1, cmap='Blues', values_format='d'
    )
    ax1.set_title('Random Forest', fontsize=14)
    
    # SVM Confusion Matrix
    ConfusionMatrixDisplay(svm_cm, display_labels=['Normal', 'Faulty']).plot(
        ax=ax2, cmap='Oranges', values_format='d'
    )
    ax2.set_title('SVM', fontsize=14)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to: {save_path}")
    plt.show()


def main():
    """
    Main function to run the equipment anomaly detection analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Equipment Anomaly Detection')
    parser.add_argument('--model', choices=['rf', 'svm', 'all'], default='all',
                       help='Model to run: rf (Random Forest), svm (SVM), or all')
    parser.add_argument('--data', default='equipment_anomaly_data.csv',
                       help='Path to the dataset CSV file')
    
    args = parser.parse_args()
    
    print("🔧 EQUIPMENT ANOMALY DETECTION SYSTEM")
    print(f"📊 Running model(s): {args.model.upper()}")
    print(f"📁 Using dataset: {args.data}")
    
    # Load and preprocess data
    df = load_and_explore_data(args.data)
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Initialize results storage
    results = {}
    
    # Train and evaluate models based on user selection
    if args.model in ['rf', 'all']:
        print("\n" + "=" * 60)
        print("RANDOM FOREST ANALYSIS")
        print("=" * 60)
        
        rf_model = train_random_forest(X_train, y_train)
        rf_predictions, rf_report = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        
        results['rf'] = {
            'model': rf_model,
            'predictions': rf_predictions,
            'report': rf_report,
            'accuracy': rf_accuracy
        }
    
    if args.model in ['svm', 'all']:
        print("\n" + "=" * 60)
        print("SVM ANALYSIS")
        print("=" * 60)
        
        svm_model = train_svm(X_train, y_train)
        svm_predictions, svm_report = evaluate_model(svm_model, X_test, y_test, "SVM")
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        
        results['svm'] = {
            'model': svm_model,
            'predictions': svm_predictions,
            'report': svm_report,
            'accuracy': svm_accuracy
        }
    
    # Create comparisons if both models were run
    if len(results) == 2:
        create_comparison_table(
            results['rf']['report'], results['svm']['report'],
            results['rf']['accuracy'], results['svm']['accuracy']
        )
        
        plot_metrics_comparison(results['rf']['report'], results['svm']['report'])
        plot_confusion_matrices(y_test, results['rf']['predictions'], results['svm']['predictions'])
        
        # Provide recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        if results['svm']['accuracy'] > results['rf']['accuracy']:
            print("🏆 SVM shows better overall performance")
            print("💡 Recommended for critical biomedical equipment monitoring")
            print("   where false negatives (missed faults) are unacceptable")
        else:
            print("🏆 Random Forest shows better overall performance")
            print("💡 Recommended for diagnostic dashboards where feature")
            print("   importance interpretation is valuable")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("📈 Check the 'images/' directory for visualizations")
    print("🔍 Review the metrics above for model selection guidance")


if __name__ == "__main__":
    main()