#!/usr/bin/env python3
"""
Aggregate experiment results into a master Excel file.
Appends new results to existing experiments spreadsheet.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Aggregate results to master Excel')
    parser.add_argument('--input', type=str, required=True, help='Path to new results Excel')
    parser.add_argument('--master', type=str, required=True, help='Path to master Excel file')
    args = parser.parse_args()
    
    master_path = Path(args.master)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load new results
    print(f"📂 Loading new results from: {args.input}")
    new_df = pd.read_excel(args.input)
    
    # Load or create master file
    if master_path.exists():
        print(f"📊 Loading existing master file: {master_path}")
        master_df = pd.read_excel(master_path)
        
        # Check for duplicate artifact_id
        if 'artifact_id' in new_df.columns and 'artifact_id' in master_df.columns:
            new_ids = set(new_df['artifact_id'].tolist())
            existing_ids = set(master_df['artifact_id'].tolist())
            duplicates = new_ids & existing_ids
            
            if duplicates:
                print(f"⚠️ Skipping duplicate artifact IDs: {duplicates}")
                new_df = new_df[~new_df['artifact_id'].isin(duplicates)]
        
        if len(new_df) > 0:
            # Append new results
            master_df = pd.concat([master_df, new_df], ignore_index=True)
            print(f"➕ Added {len(new_df)} new experiment(s)")
        else:
            print("ℹ️ No new experiments to add")
    else:
        print(f"📝 Creating new master file: {master_path}")
        master_df = new_df
    
    # Sort by timestamp (most recent first)
    if 'timestamp' in master_df.columns:
        master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
        master_df = master_df.sort_values('timestamp', ascending=False)
    
    # Save master file
    print(f"💾 Saving master file with {len(master_df)} total experiments")
    
    with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
        master_df.to_excel(writer, sheet_name='All Experiments', index=False)
        
        # Create summary sheet
        if len(master_df) > 0:
            # Get unique domains
            source_domains = ', '.join(master_df['source_domain'].unique()) if 'source_domain' in master_df.columns else 'N/A'
            target_domains = ', '.join(master_df['target_domain'].unique()) if 'target_domain' in master_df.columns else 'N/A'
            
            summary_data = {
                'Total Experiments': len(master_df),
                'Models Used': ', '.join(master_df['model_type'].unique()) if 'model_type' in master_df.columns else 'N/A',
                'Source Domains': source_domains,
                'Target Domains': target_domains,
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            # Best results per metric
            metric_cols = [c for c in master_df.columns if any(m in c for m in ['Recall', 'NDCG', 'HR', 'MRR'])]
            for col in metric_cols:
                if col in master_df.columns and master_df[col].notna().any():
                    best_idx = master_df[col].idxmax()
                    best_val = master_df.loc[best_idx, col]
                    best_exp = master_df.loc[best_idx, 'experiment_name'] if 'experiment_name' in master_df.columns else 'Unknown'
                    summary_data[f'Best {col}'] = f"{best_val:.4f} ({best_exp})"
            
            summary_df = pd.DataFrame([summary_data]).T.reset_index()
            summary_df.columns = ['Metric', 'Value']
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format columns
        worksheet = writer.sheets['All Experiments']
        for idx, col in enumerate(master_df.columns, 1):
            max_length = max(
                master_df[col].astype(str).map(len).max() if len(master_df) > 0 else 0,
                len(col)
            ) + 2
            col_letter = chr(64 + idx) if idx <= 26 else 'A' + chr(64 + idx - 26)
            worksheet.column_dimensions[col_letter].width = min(max_length, 30)
    
    print(f"✅ Master file updated successfully!")


if __name__ == '__main__':
    main()
