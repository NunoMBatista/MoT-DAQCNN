#!/usr/bin/env python3
"""
Batch Runner — Execute multiple commands sequentially with logging
=================================================================

This script runs multiple shell commands in sequence, logging their progress
and results. Useful for running multiple hyperparameter searches, experiments,
or any batch of long-running tasks.

Usage:
    # Run from a commands file:
    python batch_runner.py commands.txt

    # Or specify commands inline:
    python batch_runner.py \
        "python train.py --config config1.yml" \
        "python train.py --config config2.yml"

Commands file format (one command per line):
    python experiments/hyperparameter_search.py --config configs/breast_mnist_ts_moe_3kern.yml --search-config configs/hp_search/breast_mnist_ts_moe.yml --n-trials 80
    python experiments/hyperparameter_search.py --config configs/breast_mnist_gumbel_3kern.yml --search-config configs/hp_search/breast_mnist_gumbel.yml --n-trials 80
    # Lines starting with # are ignored

Features:
    - Runs commands sequentially (one at a time)
    - Logs stdout/stderr to individual files
    - Tracks elapsed time per command
    - Optionally continues on errors (--continue-on-error)
    - Saves summary to batch_summary.txt
"""

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: str, log_dir: Path, cmd_idx: int, continue_on_error: bool):
    """
    Run a single command and log its output.
    
    Returns:
        (success: bool, elapsed_time: float)
    """
    print(f"\n{'='*80}")
    print(f"[{cmd_idx}] Running: {cmd}")
    print(f"{'='*80}")
    
    # Ensure log directory exists
    if not log_dir.exists():
        print(f"ERROR: Log directory does not exist: {log_dir}")
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Create log files
    log_file = log_dir / f"cmd_{cmd_idx:03d}.log"
    err_file = log_dir / f"cmd_{cmd_idx:03d}.err"
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log_f, open(err_file, 'w') as err_f:
            # Write command header
            log_f.write(f"Command: {cmd}\n")
            log_f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"{'-'*80}\n\n")
            log_f.flush()
            
            # Run command
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Use threads to read stdout and stderr concurrently (avoids deadlock)
            def read_stdout():
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()
            
            def read_stderr():
                for line in process.stderr:
                    print(line, end='', file=sys.stderr)
                    err_f.write(line)
                    err_f.flush()
            
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for threads to finish reading
            stdout_thread.join()
            stderr_thread.join()
            
        elapsed = time.time() - start_time
        
        if return_code == 0:
            print(f"\n✓ Command completed successfully in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n✗ Command failed with exit code {return_code} after {elapsed:.1f}s")
            if not continue_on_error:
                print(f"  Stopping batch execution. Check logs in {log_dir}")
                print(f"  Stdout: {log_file}")
                print(f"  Stderr: {err_file}")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Exception while running command: {e}")
        if not continue_on_error:
            print(f"  Stopping batch execution. Check logs in {log_dir}")
        return False, elapsed


def load_commands(source: str) -> list[str]:
    """Load commands from a file or return as a single command."""
    path = Path(source)
    
    if path.is_file():
        # Load from file
        with open(path, 'r') as f:
            commands = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    commands.append(line)
            return commands
    else:
        # Treat as a single command
        return [source]


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple commands sequentially with logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'commands',
        nargs='+',
        help='Commands file (one per line) or inline commands'
    )
    parser.add_argument(
        '--log-dir',
        default='batch_logs',
        help='Directory to store logs (default: batch_logs)'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue running commands even if one fails'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    
    args = parser.parse_args()
    
    # Load all commands
    all_commands = []
    for source in args.commands:
        all_commands.extend(load_commands(source))
    
    if not all_commands:
        print("No commands to run!")
        return 1
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"run_{timestamp}"
    log_dir = log_dir.resolve()  # Convert to absolute path
    
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs will be saved to: {log_dir}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Batch Runner — {len(all_commands)} command(s) to execute")
    print(f"{'='*80}")
    for idx, cmd in enumerate(all_commands, 1):
        print(f"  [{idx}] {cmd}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("Dry run — not executing commands")
        return 0
    
    # Execute commands
    results = []
    batch_start = time.time()
    
    for idx, cmd in enumerate(all_commands, 1):
        success, elapsed = run_command(cmd, log_dir, idx, args.continue_on_error)
        results.append({
            'cmd': cmd,
            'idx': idx,
            'success': success,
            'elapsed': elapsed
        })
        
        if not success and not args.continue_on_error:
            break
    
    batch_elapsed = time.time() - batch_start
    
    # Write summary
    summary_file = log_dir / "batch_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Batch Execution Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Started:  {datetime.fromtimestamp(batch_start).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {batch_elapsed:.1f}s ({batch_elapsed/60:.1f} min)\n\n")
        
        f.write(f"Results:\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
            f.write(f"[{r['idx']:3d}] {status:10s} ({r['elapsed']:7.1f}s) — {r['cmd']}\n")
        
        f.write("\n")
        success_count = sum(1 for r in results if r['success'])
        f.write(f"Summary: {success_count}/{len(results)} succeeded\n")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("Batch Execution Complete")
    print(f"{'='*80}")
    print(f"Total time: {batch_elapsed:.1f}s ({batch_elapsed/60:.1f} min)")
    print(f"Results: {success_count}/{len(results)} succeeded")
    print(f"\nLogs saved to: {log_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}\n")
    
    # Return exit code based on success
    return 0 if success_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
