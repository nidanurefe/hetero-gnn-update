#!/usr/bin/env python3
"""
Local pipeline trigger - Triggers GitHub Actions workflow locally.
Or runs directly on remote server via SSH.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_local_ssh_pipeline(args):
    """Start training directly on remote server via SSH."""
    
    script_path = Path(__file__).parent / "remote_train.sh"
    
    cmd = [
        "bash", str(script_path),
        "--host", args.host,
        "--user", args.user,
        "--path", args.remote_path,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden", str(args.hidden),
        "--neg-ratio", str(args.neg_ratio),
        "--exp-name", args.exp_name,
        "--output-dir", args.output_dir,
    ]
    
    print(f"🚀 Starting local SSH pipeline...")
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def trigger_github_workflow(args):
    """Trigger GitHub Actions workflow."""
    
    # Check if gh CLI is installed
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ GitHub CLI (gh) is not installed!")
        print("   Install it with: brew install gh")
        print("   Then authenticate: gh auth login")
        return 1
    
    # Trigger workflow
    cmd = [
        "gh", "workflow", "run", "gnn-pipeline.yml",
        "-f", f"dataset={args.dataset}",
        "-f", f"model_type={args.model}",
        "-f", f"epochs={args.epochs}",
        "-f", f"batch_size={args.batch_size}",
        "-f", f"learning_rate={args.lr}",
        "-f", f"hidden_dim={args.hidden}",
        "-f", f"neg_ratio={args.neg_ratio}",
        "-f", f"experiment_name={args.exp_name}",
    ]
    
    print(f"🚀 Triggering GitHub Actions workflow...")
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Workflow triggered successfully!")
        print("   View progress at: gh run list --workflow=gnn-pipeline.yml")
        print("   Or visit: https://github.com/<your-repo>/actions")
    
    return result.returncode


def run_evaluation(args):
    """Run evaluation locally and write results to Excel."""
    
    project_root = Path(__file__).parent.parent
    
    # Create evaluation config
    config_content = f"""
data:
  proc_dir: data/processed

model:
  name: {args.model}
  ckpt_path: {args.output_dir}/{args.exp_name}_{args.model}.pt
  hidden: {args.hidden}
  heads: 4

eval:
  ks: [5, 10, 20]
  num_neg: 99
  seed: 42
"""
    
    config_path = project_root / "config" / "config_eval_local.yaml"
    config_path.write_text(config_content)
    
    print(f"\n📊 Running evaluation...")
    
    # Evaluation çalıştır
    eval_cmd = ["python", "src/eval_rank.py", "--config", str(config_path)]
    eval_log_path = Path(args.output_dir) / "evaluation.log"
    
    with open(eval_log_path, 'w') as log_file:
        result = subprocess.run(eval_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_file.write(result.stdout)
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed! Check {eval_log_path}")
        return result.returncode
    
    # Sonuçları Excel'e yaz
    print(f"\n📈 Exporting results to Excel...")
    
    params_path = Path(args.output_dir) / "params.json"
    excel_path = Path(args.output_dir) / f"{args.exp_name}_results.xlsx"
    artifact_id = f"{args.exp_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    export_cmd = [
        "python", "scripts/export_results_to_excel.py",
        "--params", str(params_path),
        "--eval-log", str(eval_log_path),
        "--output", str(excel_path),
        "--artifact-id", artifact_id,
    ]
    
    result = subprocess.run(export_cmd, cwd=project_root)
    
    if result.returncode == 0:
        print(f"\n✅ Results saved to: {excel_path}")
        
        # Add to master Excel
        master_path = project_root / "results" / "all_experiments.xlsx"
        agg_cmd = [
            "python", "scripts/aggregate_results.py",
            "--input", str(excel_path),
            "--master", str(master_path),
        ]
        subprocess.run(agg_cmd, cwd=project_root)
        print(f"📊 Results aggregated to: {master_path}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='GNN Training Pipeline Trigger',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training directly on remote server via SSH
  python trigger_pipeline.py ssh --exp-name exp001 --model gat --epochs 50

  # Trigger GitHub Actions workflow
  python trigger_pipeline.py github --exp-name exp001 --model gat --epochs 50

  # Run local evaluation
  python trigger_pipeline.py eval --exp-name exp001 --output-dir ./results
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Pipeline mode')
    
    # SSH mode
    ssh_parser = subparsers.add_parser('ssh', help='Run via direct SSH')
    ssh_parser.add_argument('--host', type=str, default=os.getenv('REMOTE_HOST'), help='SSH host')
    ssh_parser.add_argument('--user', type=str, default=os.getenv('REMOTE_USER'), help='SSH user')
    ssh_parser.add_argument('--remote-path', type=str, default=os.getenv('REMOTE_PATH'), help='Remote project path')
    
    # GitHub mode
    gh_parser = subparsers.add_parser('github', help='Trigger GitHub Actions workflow')
    gh_parser.add_argument('--dataset', type=str, default='amazon_movies', help='Dataset name')
    
    # Eval mode
    eval_parser = subparsers.add_parser('eval', help='Run local evaluation')
    
    # Common arguments for all modes
    for p in [ssh_parser, gh_parser, eval_parser]:
        p.add_argument('--exp-name', type=str, required=True, help='Experiment name')
        p.add_argument('--model', type=str, default='gat', choices=['gat', 'sage'], help='Model type')
        p.add_argument('--epochs', type=int, default=30, help='Training epochs')
        p.add_argument('--batch-size', type=int, default=4096, help='Batch size')
        p.add_argument('--lr', type=float, default=0.002, help='Learning rate')
        p.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
        p.add_argument('--neg-ratio', type=float, default=5.0, help='Negative sample ratio')
        p.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    if args.mode == 'ssh':
        if not all([args.host, args.user, args.remote_path]):
            print("❌ SSH mode requires --host, --user, and --remote-path")
            print("   Or set REMOTE_HOST, REMOTE_USER, REMOTE_PATH environment variables")
            return 1
        return run_local_ssh_pipeline(args)
    
    elif args.mode == 'github':
        return trigger_github_workflow(args)
    
    elif args.mode == 'eval':
        return run_evaluation(args)


if __name__ == '__main__':
    sys.exit(main())
