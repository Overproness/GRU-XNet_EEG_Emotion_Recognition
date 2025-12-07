"""
Apply augmentation to all available datasets
Processes DEAP, GAMEEMO, and SEED-IV datasets and saves augmented versions
"""

import numpy as np
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from augmentation_config import AugmentationConfig
from augmentation_pipeline import AugmentationPipeline
from dataset_loaders import get_loader
from quality_validation import DataQualityValidator


def augment_deap(base_dir: str, output_dir: str, config: AugmentationConfig):
    """
    Augment DEAP dataset
    
    Args:
        base_dir: Base directory containing datasets
        output_dir: Output directory for augmented data
        config: Augmentation configuration
    """
    print("\n" + "="*100)
    print("AUGMENTING DEAP DATASET")
    print("="*100)
    
    deap_path = os.path.join(base_dir, "datasets", "deap")
    loader = get_loader("DEAP", deap_path)
    
    # Update config for DEAP
    config.dataset_sampling_rates['DEAP'] = 128
    
    # Create output directory
    deap_output = Path(output_dir) / "DEAP_augmented"
    deap_output.mkdir(parents=True, exist_ok=True)
    
    # Option 1: Augment all subjects together
    print("\nOption: Augment all subjects")
    all_data, all_labels, subject_ids = loader.load_all_subjects()
    
    print(f"Loaded {len(all_data)} total trials from {len(set(subject_ids))} subjects")
    
    # Create binary labels for valence
    binary_labels = loader.create_binary_labels(all_labels, 'valence', threshold=5.0)
    
    print(f"Binary labels: {np.sum(binary_labels)} high valence, {len(binary_labels) - np.sum(binary_labels)} low valence")
    
    # Augment
    pipeline = AugmentationPipeline(config)
    aug_data, aug_labels = pipeline.augment_dataset(all_data, binary_labels)
    
    # Save
    pipeline.save_augmented_dataset(aug_data, aug_labels, deap_output, "deap_all_subjects")
    
    # Also save subject IDs (extended for augmented samples)
    n_original = len(all_data)
    n_augmented = len(aug_data) - n_original
    
    # Randomly assign augmented samples to subjects proportionally
    aug_subject_ids = np.concatenate([
        subject_ids,  # Original subject IDs
        np.random.choice(subject_ids, n_augmented)  # Random assignment for augmented
    ])
    
    np.save(deap_output / "deap_all_subjects_subject_ids.npy", aug_subject_ids)
    
    # Quality validation
    print("\nValidating quality...")
    validator = DataQualityValidator()
    batch_results = validator.validate_batch(
        all_data[:20],
        aug_data[n_original:n_original+20],
        fs=128,
        n_samples=20
    )
    validator.print_validation_report(batch_results)
    
    pipeline.print_statistics()
    
    print(f"\n   DEAP augmentation completed!")
    print(f"  Output directory: {deap_output}")
    
    return True


def augment_deap_per_subject(base_dir: str, output_dir: str, config: AugmentationConfig):
    """
    Augment DEAP dataset subject by subject (for LOSO cross-validation)
    
    Args:
        base_dir: Base directory containing datasets
        output_dir: Output directory for augmented data
        config: Augmentation configuration
    """
    print("\n" + "="*100)
    print("AUGMENTING DEAP DATASET (PER SUBJECT)")
    print("="*100)
    
    deap_path = os.path.join(base_dir, "datasets", "deap")
    loader = get_loader("DEAP", deap_path)
    
    # Update config for DEAP
    config.dataset_sampling_rates['DEAP'] = 128
    
    # Create output directory
    deap_output = Path(output_dir) / "DEAP_augmented_per_subject"
    deap_output.mkdir(parents=True, exist_ok=True)
    
    # Augment each subject separately
    for subject_id in range(1, 33):  # 32 subjects
        print(f"\n{'='*80}")
        print(f"Subject {subject_id}/32")
        print(f"{'='*80}")
        
        try:
            # Load subject
            data, labels = loader.load_subject(subject_id)
            binary_labels = loader.create_binary_labels(labels, 'valence', threshold=5.0)
            
            print(f"Loaded: {data.shape[0]} trials, {data.shape[1]} channels")
            
            # Augment
            pipeline = AugmentationPipeline(config)
            aug_data, aug_labels = pipeline.augment_dataset(data, binary_labels)
            
            # Save
            subject_output = deap_output / f"subject_{subject_id:02d}"
            subject_output.mkdir(exist_ok=True)
            
            pipeline.save_augmented_dataset(
                aug_data, aug_labels, 
                subject_output, 
                f"deap_subject_{subject_id:02d}"
            )
            
            print(f"   Subject {subject_id} completed: {len(aug_data)} total samples")
            
        except FileNotFoundError:
            print(f"✗ Subject {subject_id} data not found, skipping")
            continue
        except Exception as e:
            print(f"✗ Error processing subject {subject_id}: {e}")
            continue
    
    print(f"\n   All subjects augmented!")
    print(f"  Output directory: {deap_output}")
    
    return True


def augment_gameemo(base_dir: str, output_dir: str, config: AugmentationConfig):
    """
    Augment GAMEEMO dataset
    
    Args:
        base_dir: Base directory containing datasets
        output_dir: Output directory for augmented data
        config: Augmentation configuration
    """
    print("\n" + "="*100)
    print("AUGMENTING GAMEEMO DATASET")
    print("="*100)
    
    gameemo_path = os.path.join(base_dir, "datasets", "GAMEEMO")
    
    # Update config for GAMEEMO
    config.dataset_sampling_rates['GAMEEMO'] = 128
    
    # Create output directory
    gameemo_output = Path(output_dir) / "GAMEEMO_augmented"
    gameemo_output.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading GAMEEMO dataset...")
    try:
        loader = get_loader("GAMEEMO", gameemo_path)
        all_data, all_labels, subject_ids = loader.load_all_subjects(use_binary_labels=True)
        
        print(f"\nLoaded {len(all_data)} total windows from {len(set(subject_ids))} subjects")
        print(f"Data shape: {all_data.shape}")
        print(f"Binary labels: {np.sum(all_labels)} high arousal, {len(all_labels) - np.sum(all_labels)} low arousal")
        
        # Augment
        print("\nAugmenting data...")
        pipeline = AugmentationPipeline(config)
        aug_data, aug_labels = pipeline.augment_dataset(all_data, all_labels)
        
        # Save
        print("\nSaving augmented dataset...")
        pipeline.save_augmented_dataset(aug_data, aug_labels, gameemo_output, "gameemo_all_subjects")
        
        # Also save subject IDs (extended for augmented samples)
        n_original = len(all_data)
        n_augmented = len(aug_data) - n_original
        
        # Randomly assign augmented samples to subjects proportionally
        aug_subject_ids = np.concatenate([
            subject_ids,
            np.random.choice(subject_ids, n_augmented)
        ])
        
        np.save(gameemo_output / "gameemo_all_subjects_subject_ids.npy", aug_subject_ids)
        
        # Quality validation
        print("\nValidating quality...")
        validator = DataQualityValidator()
        batch_results = validator.validate_batch(
            all_data[:20],
            aug_data[n_original:n_original+20],
            fs=128,
            n_samples=20
        )
        validator.print_validation_report(batch_results)
        
        pipeline.print_statistics()
        
        print(f"\n   GAMEEMO augmentation completed!")
        print(f"  Output directory: {gameemo_output}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ GAMEEMO dataset not found: {e}")
        print("  Please ensure the GAMEEMO dataset is in the correct location.")
        return False
    except Exception as e:
        print(f"\n✗ GAMEEMO augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def augment_seediv(base_dir: str, output_dir: str, config: AugmentationConfig):
    """
    Augment SEED-IV dataset
    
    Args:
        base_dir: Base directory containing datasets
        output_dir: Output directory for augmented data
        config: Augmentation configuration
    """
    print("\n" + "="*100)
    print("AUGMENTING SEED-IV DATASET")
    print("="*100)
    
    seediv_path = os.path.join(base_dir, "datasets", "sead-4")
    
    try:
        loader = get_loader("SEED-IV", seediv_path)
    except ImportError as e:
        print(f"\n    Error: {e}")
        print("\nTo install scipy, run: pip install scipy")
        return False
    
    # Update config for SEED-IV
    config.dataset_sampling_rates['SEED-IV'] = 200
    
    # Create output directory
    seediv_output = Path(output_dir) / "SEEDIV_augmented"
    seediv_output.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading SEED-IV dataset...")
    try:
        all_data, all_labels, subject_ids = loader.load_all_subjects()
        
        if all_data is None:
            print("    Failed to load data")
            return False
        
        print(f"\nLoaded {len(all_data)} total trials from {len(set(subject_ids))} subjects")
        print(f"Data shape: {all_data.shape}")
        print(f"Labels distribution: {np.bincount(all_labels)}")
        print(f"  0=neutral: {np.sum(all_labels == 0)}")
        print(f"  1=sad: {np.sum(all_labels == 1)}")
        print(f"  2=fear: {np.sum(all_labels == 2)}")
        print(f"  3=happy: {np.sum(all_labels == 3)}")
        
        # Augment
        print("\nAugmenting dataset...")
        pipeline = AugmentationPipeline(config)
        aug_data, aug_labels = pipeline.augment_dataset(all_data, all_labels)
        
        # Save
        pipeline.save_augmented_dataset(aug_data, aug_labels, seediv_output, "seediv_all_subjects")
        
        # Also save subject IDs (extended for augmented samples)
        n_original = len(all_data)
        n_augmented = len(aug_data) - n_original
        
        aug_subject_ids = np.concatenate([
            subject_ids,
            np.random.choice(subject_ids, n_augmented)
        ])
        
        np.save(seediv_output / "seediv_all_subjects_subject_ids.npy", aug_subject_ids)
        
        # Quality validation
        print("\nValidating quality...")
        validator = DataQualityValidator()
        batch_results = validator.validate_batch(
            all_data[:20],
            aug_data[n_original:n_original+20],
            fs=200,
            n_samples=20
        )
        validator.print_validation_report(batch_results)
        
        pipeline.print_statistics()
        
        print(f"\n   SEED-IV augmentation completed!")
        print(f"  Output directory: {seediv_output}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n    Error: {e}")
        print(f"\nPlease ensure SEED-IV data is in: {seediv_path}")
        return False
    except Exception as e:
        print(f"\n    Error during augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with command-line arguments"""
    parser = argparse.ArgumentParser(description='Augment EEG datasets')
    
    parser.add_argument('--dataset', type=str, default='DEAP',
                       choices=['DEAP', 'DEAP_per_subject', 'SEEDIV', 'GAMEEMO', 'all'],
                       help='Dataset to augment')
    
    parser.add_argument('--base_dir', type=str,
                       default=r'',
                       help='Base directory containing datasets folder')
    
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory (default: OurApproach/augmented_datasets)')
    
    parser.add_argument('--ratio', type=float, default=2.0,
                       help='Augmentation ratio (default: 2.0 for 1:2)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "OurApproach", "augmented_datasets")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = AugmentationConfig()
    config.target_augmentation_ratio = args.ratio
    config.random_seed = args.seed
    
    print("="*100)
    print("EEG DATA AUGMENTATION")
    print("="*100)
    print(f"Dataset: {args.dataset}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentation ratio: 1:{args.ratio}")
    print(f"Random seed: {args.seed}")
    print("="*100)
    
    # Print configuration
    config.print_summary()
    
    # Augment selected dataset(s)
    success = True
    
    if args.dataset == 'DEAP' or args.dataset == 'all':
        try:
            augment_deap(args.base_dir, args.output_dir, config)
        except Exception as e:
            print(f"\n✗ DEAP augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    if args.dataset == 'DEAP_per_subject':
        try:
            augment_deap_per_subject(args.base_dir, args.output_dir, config)
        except Exception as e:
            print(f"\n✗ DEAP per-subject augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    if args.dataset == 'GAMEEMO' or args.dataset == 'all':
        try:
            success = augment_gameemo(args.base_dir, args.output_dir, config) and success
        except Exception as e:
            print(f"\n✗ GAMEEMO augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    if args.dataset == 'SEEDIV' or args.dataset == 'all':
        augment_seediv(args.base_dir, args.output_dir, config)
    
    # Summary
    print("\n" + "="*100)
    if success:
        print("   AUGMENTATION COMPLETED SUCCESSFULLY!")
    else:
        print("    AUGMENTATION COMPLETED WITH ERRORS")
    print(f"Results saved to: {args.output_dir}")
    print("="*100)


if __name__ == "__main__":
    main()
