#!/usr/bin/env python3
"""
Comprehensive test script for all DTI model variants and configurations.
"""

import sys
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestConfiguration:
    """Represents a single test configuration."""
    
    def __init__(self, model: str, phase: str, dataset: str = None, split: str = None, 
                 pretrain_target: str = None, description: str = None):
        self.model = model
        self.phase = phase  
        self.dataset = dataset
        self.split = split
        self.pretrain_target = pretrain_target
        self.description = description or f"{model}_{phase}_{dataset}_{split}_{pretrain_target}"
        
    def to_command_args(self) -> List[str]:
        """Convert to command line arguments for run.py"""
        args = [
            "--model", self.model,
            "--phase", self.phase,
        ]
        
        if self.dataset:
            args.extend(["--dataset", self.dataset])
        if self.split:
            args.extend(["--split", self.split])
        if self.pretrain_target:
            args.extend(["--pretrain_target", self.pretrain_target])
            
        return args

class DTITester:
    """Main testing class that orchestrates all model tests."""
    
    def __init__(self, debug_mode: bool = True, python_executable: str = "python"):
        self.debug_mode = debug_mode
        self.python_executable = python_executable
        self.run_script = Path(__file__).parent / "run.py"
        self.results = []
        
    def get_baseline_configurations(self) -> List[TestConfiguration]:
        """Get baseline model test configurations."""
        configs = []
        
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["cold", "rand"]:
                configs.append(TestConfiguration(
                    model="baseline",
                    phase="finetune", 
                    dataset=dataset,
                    split=split,
                    description=f"Baseline: {dataset} {split} split"
                ))
        
        return configs
    
    def get_multi_modal_configurations(self) -> List[TestConfiguration]:
        """Get multi-modal model test configurations."""
        configs = []
        
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["cold", "rand"]:
                configs.append(TestConfiguration(
                    model="multi_modal",
                    phase="finetune",
                    dataset=dataset, 
                    split=split,
                    description=f"Multi-modal: {dataset} {split} split"
                ))
        
        return configs
    
    def get_multi_output_configurations(self) -> List[TestConfiguration]:
        """Get multi-output model test configurations."""
        configs = []
        
        # Training phase
        for split in ["cold", "rand"]:
            configs.append(TestConfiguration(
                model="multi_output",
                phase="train",
                split=split,
                description=f"Multi-output: train {split} split"
            ))
        
        # Finetuning phase  
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["cold", "rand"]:
                configs.append(TestConfiguration(
                    model="multi_output",
                    phase="finetune",
                    dataset=dataset,
                    split=split,
                    description=f"Multi-output: finetune {dataset} {split} split"
                ))
        
        return configs
    
    def get_multi_hybrid_configurations(self) -> List[TestConfiguration]:
        """Get multi-hybrid model test configurations."""
        configs = []
        
        # Pretraining phases
        for pretrain_target in ["drug", "target"]:
            configs.append(TestConfiguration(
                model="multi_hybrid", 
                phase="pretrain",
                pretrain_target=pretrain_target,
                description=f"Multi-hybrid: pretrain {pretrain_target}"
            ))
        
        # Training phase
        for split in ["cold", "rand"]:
            configs.append(TestConfiguration(
                model="multi_hybrid",
                phase="train", 
                split=split,
                description=f"Multi-hybrid: train {split} split"
            ))
        
        # Finetuning phase
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["cold", "rand"]:
                configs.append(TestConfiguration(
                    model="multi_hybrid",
                    phase="finetune",
                    dataset=dataset,
                    split=split,
                    description=f"Multi-hybrid: finetune {dataset} {split} split"
                ))
        
        return configs
    
    def get_all_configurations(self, model_filter: str = None, phase_filter: str = None) -> List[TestConfiguration]:
        """Get all test configurations, optionally filtered."""
        all_configs = []
        
        if model_filter is None or model_filter == "baseline":
            all_configs.extend(self.get_baseline_configurations())
        
        if model_filter is None or model_filter == "multi_modal":
            all_configs.extend(self.get_multi_modal_configurations())
            
        if model_filter is None or model_filter == "multi_output":
            all_configs.extend(self.get_multi_output_configurations())
            
        if model_filter is None or model_filter == "multi_hybrid":
            all_configs.extend(self.get_multi_hybrid_configurations())
        
        # Apply phase filter
        if phase_filter:
            all_configs = [c for c in all_configs if c.phase == phase_filter]
            
        return all_configs
    
    def run_single_test(self, config: TestConfiguration) -> Dict:
        """Run a single test configuration and return results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config.description}")
        logger.info(f"{'='*60}")
        
        # Build command
        cmd_args = [self.python_executable, str(self.run_script)]
        cmd_args.extend(config.to_command_args())
        
        if self.debug_mode:
            cmd_args.append("--debug")
            
        # Add additional overrides for quick testing
        cmd_args.extend([
            "--override",
            "hardware.gpus=0",  # Force CPU for testing
            "hardware.deterministic=false",
            "data.pin_memory=false", 
            "data.num_workers=0",
            "logging.log_every_n_steps=1"
        ])
        
        logger.info(f"Running command: {' '.join(cmd_args)}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the command
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse output for key information
            stdout_lines = result.stdout.split('\n')
            stderr_lines = result.stderr.split('\n')
            
            # Extract metrics
            num_params = self._extract_parameter_count(stdout_lines)
            metrics = self._extract_metrics(stdout_lines)
            errors = self._extract_errors(stderr_lines)
            
            # Determine if test passed
            test_passed = (result.returncode == 0 and 
                          len([e for e in errors if 'error' in e.lower()]) == 0)
            
            test_result = {
                'config': config.description,
                'model': config.model,
                'phase': config.phase,
                'dataset': config.dataset,
                'split': config.split,
                'pretrain_target': config.pretrain_target,
                'passed': test_passed,
                'return_code': result.returncode,
                'duration': duration,
                'num_parameters': num_params,
                'metrics': metrics,
                'errors': errors,
                'stdout_sample': '\n'.join(stdout_lines[-20:]),  # Last 20 lines
                'stderr_sample': '\n'.join(stderr_lines[-10:]) if stderr_lines else ''  # Last 10 lines
            }
            
            if test_passed:
                logger.info(f"✅ Test PASSED ({duration:.1f}s)")
                if num_params:
                    logger.info(f"   Model parameters: {num_params:,}")
                if metrics:
                    logger.info(f"   Final metrics: {metrics}")
            else:
                logger.error(f"❌ Test FAILED ({duration:.1f}s)")
                logger.error(f"   Return code: {result.returncode}")
                if errors:
                    logger.error(f"   Errors: {errors[:3]}")  # Show first 3 errors
                    
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Test TIMEOUT (>300s)")
            test_result = {
                'config': config.description,
                'model': config.model, 
                'phase': config.phase,
                'dataset': config.dataset,
                'split': config.split,
                'pretrain_target': config.pretrain_target,
                'passed': False,
                'return_code': -1,
                'duration': 300.0,
                'num_parameters': None,
                'metrics': {},
                'errors': ['Process timeout after 300 seconds'],
                'stdout_sample': '',
                'stderr_sample': ''
            }
            
        except Exception as e:
            logger.error(f"❌ Test ERROR: {e}")
            test_result = {
                'config': config.description,
                'model': config.model,
                'phase': config.phase, 
                'dataset': config.dataset,
                'split': config.split,
                'pretrain_target': config.pretrain_target,
                'passed': False,
                'return_code': -2,
                'duration': 0.0,
                'num_parameters': None,
                'metrics': {},
                'errors': [str(e)],
                'stdout_sample': '',
                'stderr_sample': ''
            }
            
        return test_result
    
    def _extract_parameter_count(self, stdout_lines: List[str]) -> Optional[int]:
        """Extract parameter count from stdout."""
        for line in stdout_lines:
            if 'parameters' in line.lower() and any(char.isdigit() for char in line):
                # Look for patterns like "1,234,567 parameters" or "Model has 1234567 trainable parameters"
                import re
                match = re.search(r'([\d,]+)\s*(?:trainable\s+)?parameters', line.lower())
                if match:
                    num_str = match.group(1).replace(',', '')
                    try:
                        return int(num_str)
                    except ValueError:
                        continue
        return None
    
    def _extract_metrics(self, stdout_lines: List[str]) -> Dict:
        """Extract validation/test metrics from stdout."""
        metrics = {}
        
        for line in stdout_lines:
            # Look for validation metrics
            if 'val_loss' in line.lower():
                import re
                # Pattern like "val_loss=0.1234"
                matches = re.findall(r'val_(\w+)=([0-9.-]+)', line.lower())
                for metric_name, value in matches:
                    try:
                        metrics[f'val_{metric_name}'] = float(value)
                    except ValueError:
                        continue
                        
            # Look for test metrics
            if 'test_loss' in line.lower():
                import re
                matches = re.findall(r'test_(\w+)=([0-9.-]+)', line.lower())
                for metric_name, value in matches:
                    try:
                        metrics[f'test_{metric_name}'] = float(value)
                    except ValueError:
                        continue
        
        return metrics
    
    def _extract_errors(self, stderr_lines: List[str]) -> List[str]:
        """Extract error messages from stderr."""
        errors = []
        
        for line in stderr_lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip common non-error messages
            if any(skip in line.lower() for skip in [
                'warning:', 'info:', 'debug:', 'using device', 'gpu available',
                'setting up', 'initialized'
            ]):
                continue
                
            # Look for actual errors
            if any(error_keyword in line.lower() for error_keyword in [
                'error:', 'exception:', 'traceback', 'failed:', 'could not',
                'missing', 'not found', 'invalid'
            ]):
                errors.append(line)
                
        return errors
    
    def run_all_tests(self, model_filter: str = None, phase_filter: str = None) -> None:
        """Run all tests and generate report."""
        configs = self.get_all_configurations(model_filter, phase_filter)
        
        logger.info(f"\n🚀 Starting DTI model testing")
        logger.info(f"Found {len(configs)} test configurations")
        logger.info(f"Debug mode: {self.debug_mode}")
        if model_filter:
            logger.info(f"Model filter: {model_filter}")
        if phase_filter:
            logger.info(f"Phase filter: {phase_filter}")
        
        # Run tests
        for i, config in enumerate(configs, 1):
            logger.info(f"\n📍 Progress: {i}/{len(configs)}")
            result = self.run_single_test(config)
            self.results.append(result)
            
            # Brief progress update
            passed = sum(1 for r in self.results if r['passed'])
            failed = len(self.results) - passed
            logger.info(f"   Running total: {passed} passed, {failed} failed")
        
        # Generate final report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate and display final test report."""
        logger.info(f"\n{'='*80}")
        logger.info("FINAL TEST REPORT")
        logger.info(f"{'='*80}")
        
        # Summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📊 SUMMARY")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        logger.info(f"   Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Break down by model
        models = {}
        for result in self.results:
            model = result['model']
            if model not in models:
                models[model] = {'passed': 0, 'failed': 0, 'total': 0}
            
            models[model]['total'] += 1
            if result['passed']:
                models[model]['passed'] += 1
            else:
                models[model]['failed'] += 1
        
        logger.info(f"\n📈 BY MODEL")
        for model, stats in models.items():
            pass_rate = stats['passed'] / stats['total'] * 100
            logger.info(f"   {model}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
        # Show parameter counts
        logger.info(f"\n🔢 PARAMETER COUNTS")
        for result in self.results:
            if result['passed'] and result['num_parameters']:
                logger.info(f"   {result['model']}: {result['num_parameters']:,} parameters")
        
        # List failed tests
        if failed_tests > 0:
            logger.info(f"\n❌ FAILED TESTS")
            for result in self.results:
                if not result['passed']:
                    logger.info(f"   {result['config']}")
                    if result['errors']:
                        logger.info(f"      Error: {result['errors'][0]}")
        
        # Overall status
        if failed_tests == 0:
            logger.info(f"\n🎉 ALL TESTS PASSED!")
        else:
            logger.info(f"\n⚠️  {failed_tests} TESTS FAILED")
        
        logger.info(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Test all DTI model configurations")
    
    parser.add_argument(
        "--model",
        choices=["baseline", "multi_modal", "multi_output", "multi_hybrid"],
        help="Test only specific model type"
    )
    parser.add_argument(
        "--phase", 
        choices=["pretrain", "train", "finetune"],
        help="Test only specific phase"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Run quick tests with debug mode (default)"
    )
    parser.add_argument(
        "--full",
        action="store_true", 
        help="Run full tests without debug mode"
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use"
    )
    
    args = parser.parse_args()
    
    # Determine debug mode
    debug_mode = not args.full  # Debug mode unless explicitly running full tests
    
    # Create tester
    tester = DTITester(debug_mode=debug_mode, python_executable=args.python)
    
    # Run tests
    try:
        tester.run_all_tests(model_filter=args.model, phase_filter=args.phase)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Testing interrupted by user")
        if tester.results:
            tester._generate_report()
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
