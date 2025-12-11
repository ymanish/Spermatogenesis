"""
QSSA Validation Command-Line Interface
=======================================

Command-line tool for validating QSSA in nucleosome-protamine simulations.

Usage:
    python -m src.analysis.qssa_validator.cli --help

Author: MY
Date: 2025-11-27
"""

import argparse
from pathlib import Path
import sys
import itertools

from src.core.build_nucleosomes import nucleosome_generator
from src.core.protamine import protamines

from .config import QSSAConfig
from .validation import validate_qssa_for_system
from .io import print_qssa_summary, generate_qssa_report, save_qssa_data


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate QSSA for nucleosome-protamine simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input TSV file with nucleosome free energy landscapes"
    )
    
    # Nucleosome parameters
    parser.add_argument(
        "--k-wrap",
        type=float,
        default=21.0,
        help="Nucleosome wrapping rate (1/s)"
    )
    
    parser.add_argument(
        "--binding-sites",
        type=int,
        default=14,
        help="Number of nucleosome binding sites"
    )
    
    # Protamine parameters
    parser.add_argument(
        "--prot-conc",
        type=float,
        default=100.0,
        help="Protamine concentration (μM)"
    )
    
    parser.add_argument(
        "--prot-k-bind",
        type=float,
        default=1.0,
        help="Protamine binding rate (1/(μM·s))"
    )
    
    parser.add_argument(
        "--prot-k-unbind",
        type=float,
        default=100.0,
        help="Protamine unbinding rate (1/s)"
    )
    
    parser.add_argument(
        "--prot-cooperativity",
        type=float,
        default=0.0,
        help="Protamine cooperativity parameter J (dimensionless)"
    )
    
    # QSSA parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="QSSA validity threshold for epsilon"
    )
    

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save validation reports and data"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-nucleosome details"
    )
    
    parser.add_argument(
        "--max-nucleosomes",
        type=int,
        default=None,
        help="Maximum number of nucleosomes to validate (for testing)"
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("QSSA VALIDATION")
    print("=" * 80)
    print(f"\nInput file: {args.input_file}")
    print(f"Output directory: {args.output_dir if args.output_dir else 'None (display only)'}")
    print("\nParameters:")
    print(f"  Nucleosome: k_wrap={args.k_wrap}, binding_sites={args.binding_sites}")
    print(f"  Protamine: P_conc={args.prot_conc}, k_bind={args.prot_k_bind}, "
          f"k_unbind={args.prot_k_unbind}, cooperativity={args.prot_cooperativity}")
    print(f"  QSSA: threshold={args.threshold}")
    print("=" * 80 + "\n")
    
    # Create configuration object
    config = QSSAConfig(
        k_wrap=args.k_wrap,
        binding_sites=args.binding_sites,
        prot_k_unbind=args.prot_k_unbind,
        prot_k_bind=args.prot_k_bind,
        prot_p_conc=args.prot_conc,
        prot_cooperativity=args.prot_cooperativity,
        threshold=args.threshold,
        max_nucleosomes=args.max_nucleosomes,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    # Load nucleosomes using generator
    print("Loading nucleosomes...")
    try:
        gen = nucleosome_generator(
            file_path=str(args.input_file),
            k_wrap=config.k_wrap,
            kT=config.kT,
            binding_sites=config.binding_sites
        )
        
        # Limit number of nucleosomes if specified
        if config.max_nucleosomes is not None:
            gen = itertools.islice(gen, config.max_nucleosomes)
        
        # Convert to list for validation
        nucs_list = list(gen)
        
        print(f"✓ Loaded {len(nucs_list)} nucleosomes\n")
    except Exception as e:
        print(f"✗ Error loading nucleosomes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create protamine instance
    print("Creating protamine instance...")
    try:
        prot = protamines(
            p_conc=config.prot_p_conc,
            cooperativity=config.prot_cooperativity,
            k_unbind=config.prot_k_unbind,
            k_bind=config.prot_k_bind,
        )
        print(f"✓ Created protamine instance\n")
    except Exception as e:
        print(f"✗ Error creating protamine instance: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate QSSA for each nucleosome
    print("Validating QSSA...")
    try:
        # Import here to avoid circular imports
        from .validation import QSSAValidationResult, validate_qssa_for_nucleosome
        from .core import compute_protamine_fast_timescale
        
        # Compute tau_prot once for all nucleosomes
        tau_prot = compute_protamine_fast_timescale(prot, beta=config.beta)
        
        # Validate each nucleosome
        nuc_results = []
        num_valid = 0
        num_invalid = 0
        max_eps_overall = 0.0
        
        for nuc in nucs_list:
            result = validate_qssa_for_nucleosome(nuc, tau_prot, config.threshold)
            nuc_results.append(result)
            
            if result.qssa_valid:
                num_valid += 1
            else:
                num_invalid += 1
            
            max_eps_overall = max(max_eps_overall, result.eps_max)
        
        # Build system result
        from .validation import SystemQSSAResult
        result = SystemQSSAResult(
            tau_prot=tau_prot,
            num_nucleosomes=len(nucs_list),
            num_valid=num_valid,
            num_invalid=num_invalid,
            fraction_valid=num_valid / len(nucs_list) if len(nucs_list) > 0 else 0.0,
            max_epsilon_overall=max_eps_overall,
            nucleosome_results=nuc_results,
            system_qssa_valid=(num_invalid == 0)
        )
        
        print("✓ Validation complete\n")
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print_qssa_summary(result, verbose=config.verbose)
    
    # Save results if output directory specified
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {config.output_dir}...")
        
        # Save configuration
        import json
        config_path = config.output_dir / "qssa_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"✓ Saved configuration to {config_path}")
        
        # Save report
        report_path = config.output_dir / "qssa_validation_report.txt"
        generate_qssa_report(result, report_path, include_details=True)
        
        # Save data
        data_path = config.output_dir / "qssa_validation_data.tsv"
        save_qssa_data(result, data_path)
        
        print(f"\n✓ All results saved to {config.output_dir}")
    
    print("\n" + "=" * 80)
    print("QSSA VALIDATION COMPLETE")
    print("=" * 80 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if result.system_qssa_valid else 1)


if __name__ == "__main__":
    main()
