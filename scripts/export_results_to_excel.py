#!/usr/bin/env python3
"""
Export experiment results to Excel file.
Parses evaluation log and combines with training parameters.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_evaluation_log(log_path: str) -> dict:
    """Parse evaluation metrics from log file."""
    metrics = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse Recall@K
    recall_pattern = r'Recall@(\d+):\s+([\d.]+)'
    for match in re.finditer(recall_pattern, content):
        k, value = match.groups()
        metrics[f'Recall@{k}'] = float(value)
    
    # Parse NDCG@K
    ndcg_pattern = r'NDCG@(\d+):\s+([\d.]+)'
    for match in re.finditer(ndcg_pattern, content):
        k, value = match.groups()
        metrics[f'NDCG@{k}'] = float(value)
    
    # Parse HR@K
    hr_pattern = r'HR@(\d+):\s+([\d.]+)'
    for match in re.finditer(hr_pattern, content):
        k, value = match.groups()
        metrics[f'HR@{k}'] = float(value)
    
    # Parse MRR
    mrr_pattern = r'MRR:\s+([\d.]+)'
    mrr_match = re.search(mrr_pattern, content)
    if mrr_match:
        metrics['MRR'] = float(mrr_match.group(1))
    
    # Parse number of users evaluated
    users_pattern = r'Evaluated (\d+) users'
    users_match = re.search(users_pattern, content)
    if users_match:
        metrics['evaluated_users'] = int(users_match.group(1))
    
    return metrics


def load_params(params_path: str) -> dict:
    """Load training parameters from JSON file."""
    with open(params_path, 'r') as f:
        return json.load(f)


def create_results_row(params: dict, metrics: dict, artifact_id: str) -> dict:
    """Create a single row of results combining params and metrics."""
    row = {
        # Experiment info
        'artifact_id': artifact_id,
        'experiment_name': params.get('experiment_name', ''),
        'run_id': params.get('run_id', ''),
        'timestamp': params.get('timestamp', datetime.now().isoformat()),
        
        # Model parameters
        'model_type': params.get('model_type', ''),
        'source_domain': params.get('source_domain', ''),
        'target_domain': params.get('target_domain', ''),
        'epochs': params.get('epochs', 0),
        'batch_size': params.get('batch_size', 0),
        'learning_rate': params.get('learning_rate', 0),
        'hidden_dim': params.get('hidden_dim', 0),
        'neg_ratio': params.get('neg_ratio', 0),
    }
    
    # Add all metrics
    row.update(metrics)
    
    return row


def main():
    parser = argparse.ArgumentParser(description='Export experiment results to Excel')
    parser.add_argument('--params', type=str, required=True, help='Path to params.json')
    parser.add_argument('--eval-log', type=str, required=True, help='Path to evaluation log')
    parser.add_argument('--output', type=str, required=True, help='Output Excel path')
    parser.add_argument('--artifact-id', type=str, required=True, help='Artifact ID')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📂 Loading parameters from: {args.params}")
    params = load_params(args.params)
    
    print(f"📊 Parsing evaluation log: {args.eval_log}")
    metrics = parse_evaluation_log(args.eval_log)
    
    if not metrics:
        print("⚠️ Warning: No metrics found in evaluation log!")
        sys.exit(1)
    
    print(f"📈 Found metrics: {list(metrics.keys())}")
    
    # Create results row
    row = create_results_row(params, metrics, args.artifact_id)
    
    # Create DataFrame
    df = pd.DataFrame([row])
    
    # Reorder columns for better presentation
    order = [
        'artifact_id', 'experiment_name', 'timestamp',
        'model_type', 'source_domain', 'target_domain', 'epochs', 'batch_size', 
        'learning_rate', 'hidden_dim', 'neg_ratio',
        'Recall@5', 'Recall@10', 'Recall@20',
        'NDCG@5', 'NDCG@10', 'NDCG@20',
        'HR@5', 'HR@10', 'HR@20',
        'MRR', 'evaluated_users', 'run_id'
    ]
    
    # Only keep columns that exist
    cols = [c for c in order if c in df.columns]
    remaining = [c for c in df.columns if c not in order]
    df = df[cols + remaining]
    
    # Save to Excel with formatting
    print(f"💾 Saving results to: {args.output}")
    
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Get worksheet and format
        worksheet = writer.sheets['Results']
        
        # Adjust column widths
        for idx, col in enumerate(df.columns, 1):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(64 + idx) if idx <= 26 else f'A{chr(64 + idx - 26)}'[-1]].width = min(max_length, 30)
    
    print(f"✅ Results exported successfully!")
    print(f"\n📋 Summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
