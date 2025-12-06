"""
Visualize CBSAtt model architecture
Creates a diagram showing the model structure and information flow
"""

import torch
from model import create_cbsatt_model
from torchinfo import summary


def print_model_architecture():
    """Print detailed model architecture"""
    
    print("\n" + "="*80)
    print("CBSAtt MODEL ARCHITECTURE")
    print("="*80 + "\n")
    
    # Model parameters
    n_channels = 32
    n_freq_bins = 129
    n_time_bins = 126
    n_classes = 2
    batch_size = 4
    
    # Create model
    model = create_cbsatt_model(
        n_channels=n_channels,
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        n_classes=n_classes,
        model_type='dynamic',
        gru_hidden_size=128,
        num_attention_heads=4,
        dropout=0.5
    )
    
    print("\n--- Model Summary ---\n")
    
    # Print summary
    try:
        from torchinfo import summary
        summary(
            model,
            input_size=(batch_size, n_channels, n_freq_bins, n_time_bins),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4,
            verbose=1
        )
    except ImportError:
        print("torchinfo not installed. Showing basic summary...")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n--- Parameter Count ---")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Component breakdown
    print("\n" + "="*80)
    print("COMPONENT BREAKDOWN")
    print("="*80 + "\n")
    
    print("1. Channel-Independent CNNs")
    print(f"   - Number: {n_channels}")
    print(f"   - Architecture per channel:")
    print(f"     * Conv2d(1→32, 3x3) + BN + ReLU + MaxPool(2x2)")
    print(f"     * Conv2d(32→64, 3x3) + BN + ReLU + MaxPool(2x2)")
    print(f"     * Conv2d(64→128, 3x3) + BN + ReLU + MaxPool(2x2)")
    print(f"     * Dropout(0.5)")
    print(f"   - Input per channel: (batch, 1, {n_freq_bins}, {n_time_bins})")
    print(f"   - Output per channel: (batch, 128, {n_freq_bins//8}, {n_time_bins//8})")
    
    freq_reduced = n_freq_bins // 8
    time_reduced = n_time_bins // 8
    
    print(f"\n2. BiGRU")
    print(f"   - Type: Bidirectional GRU")
    print(f"   - Layers: 2")
    print(f"   - Hidden size: 128 (256 with bidirectional)")
    print(f"   - Input: (batch, {time_reduced}, {n_channels * 128 * freq_reduced})")
    print(f"   - Output: (batch, {time_reduced}, 256)")
    
    print(f"\n3. Multi-Head Self-Attention")
    print(f"   - Number of heads: 4")
    print(f"   - Dimension per head: 64")
    print(f"   - Total dimension: 256")
    print(f"   - Mechanism: Scaled dot-product attention")
    print(f"   - Input: (batch, {time_reduced}, 256)")
    print(f"   - Output: (batch, {time_reduced}, 256)")
    
    print(f"\n4. Classification Head")
    print(f"   - Linear(256 → 256) + ReLU + Dropout")
    print(f"   - Linear(256 → 128) + ReLU + Dropout")
    print(f"   - Linear(128 → {n_classes})")
    print(f"   - Input: (batch, 256)")
    print(f"   - Output: (batch, {n_classes})")
    
    print("\n" + "="*80)
    print("DATA FLOW")
    print("="*80 + "\n")
    
    print(f"Input: (batch={batch_size}, channels={n_channels}, freq={n_freq_bins}, time={n_time_bins})")
    print(f"  ↓")
    print(f"Channel CNNs: {n_channels} independent CNNs")
    print(f"  ↓")
    print(f"Features: (batch={batch_size}, time_reduced={time_reduced}, features={n_channels * 128 * freq_reduced})")
    print(f"  ↓")
    print(f"BiGRU: (batch={batch_size}, time_reduced={time_reduced}, hidden=256)")
    print(f"  ↓")
    print(f"Attention: (batch={batch_size}, time_reduced={time_reduced}, hidden=256)")
    print(f"  ↓")
    print(f"Pooling: Global average over time dimension")
    print(f"  ↓")
    print(f"Pooled: (batch={batch_size}, hidden=256)")
    print(f"  ↓")
    print(f"Classifier: (batch={batch_size}, classes={n_classes})")
    print(f"  ↓")
    print(f"Output: (batch={batch_size}, classes={n_classes})")
    
    print("\n" + "="*80)


def compare_model_variants():
    """Compare different model variants"""
    
    print("\n" + "="*80)
    print("MODEL VARIANTS COMPARISON")
    print("="*80 + "\n")
    
    configs = [
        {'name': 'DEAP (32 channels)', 'n_channels': 32, 'n_freq_bins': 129, 'n_time_bins': 126},
        {'name': 'GAMEEMO (14 channels)', 'n_channels': 14, 'n_freq_bins': 129, 'n_time_bins': 126},
        {'name': 'SEEDIV (62 channels)', 'n_channels': 62, 'n_freq_bins': 129, 'n_time_bins': 126},
    ]
    
    print(f"{'Dataset':<25} {'Channels':<10} {'Parameters':<15} {'Memory (MB)':<12}")
    print("-" * 65)
    
    for config in configs:
        model = create_cbsatt_model(
            n_channels=config['n_channels'],
            n_freq_bins=config['n_freq_bins'],
            n_time_bins=config['n_time_bins'],
            n_classes=2,
            model_type='dynamic'
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        memory_mb = n_params * 4 / (1024**2)  # Assuming float32
        
        print(f"{config['name']:<25} {config['n_channels']:<10} {n_params:>14,} {memory_mb:>11.2f}")
    
    print("\n" + "="*80)


def print_attention_mechanism():
    """Explain attention mechanism"""
    
    print("\n" + "="*80)
    print("MULTI-HEAD SELF-ATTENTION MECHANISM")
    print("="*80 + "\n")
    
    print("Attention Formula:")
    print("  Attention(Q, K, V) = softmax(QK^T / √d_k) V")
    print()
    print("Where:")
    print("  Q = Query matrix")
    print("  K = Key matrix")
    print("  V = Value matrix")
    print("  d_k = Dimension of key vectors")
    print()
    print("Multi-Head Process:")
    print("  1. Linear projections: Q = XW_Q, K = XW_K, V = XW_V")
    print("  2. Split into h heads: (batch, seq_len, d_model) → (batch, h, seq_len, d_k)")
    print("  3. Apply attention for each head independently")
    print("  4. Concatenate heads: (batch, h, seq_len, d_k) → (batch, seq_len, d_model)")
    print("  5. Output projection: Y = MultiHead(X)W_O")
    print("  6. Residual connection: Output = LayerNorm(X + Y)")
    print()
    print("In our model:")
    print("  - Number of heads (h): 4")
    print("  - Model dimension (d_model): 256")
    print("  - Dimension per head (d_k): 256 / 4 = 64")
    print()
    print("Benefits:")
    print("  ✓ Captures different types of relationships")
    print("  ✓ Provides global context to each time step")
    print("  ✓ Dynamic feature weighting")
    print("  ✓ Interpretable attention weights")
    
    print("\n" + "="*80)


def print_preprocessing_pipeline():
    """Explain preprocessing pipeline"""
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80 + "\n")
    
    print("Step 1: Raw EEG Data")
    print("  - DEAP: (n_trials, 32, 8064) @ 128 Hz")
    print("  - GAMEEMO: (n_windows, 14, 640) @ 128 Hz")
    print("  - SEEDIV: (n_trials, 62, 28000) @ 200 Hz")
    print()
    
    print("Step 2: Short-Time Fourier Transform (STFT)")
    print("  - Window: Hann window")
    print("  - NPERSEG: Dataset-specific (128-400 samples)")
    print("  - Overlap: 50%")
    print("  - Frequency range: 0.5-50 Hz")
    print("  - Output: Time-frequency representation")
    print()
    
    print("Step 3: Standardization")
    print("  - Interpolate to common dimensions")
    print("  - Target: (n_freq_bins=129, n_time_bins=126)")
    print("  - Method: Linear interpolation")
    print()
    
    print("Step 4: Normalization (optional)")
    print("  - Method: Z-score normalization")
    print("  - Per-channel: mean=0, std=1")
    print()
    
    print("Final Output:")
    print("  All datasets: (n_samples, n_channels, 129, 126)")
    print("  Ready for model input")
    
    print("\n" + "="*80)


def main():
    """Main visualization function"""
    
    print("\n" + "🎯"*40)
    print("CBSAtt ARCHITECTURE VISUALIZATION")
    print("🎯"*40)
    
    # Print architecture
    print_model_architecture()
    
    # Compare variants
    compare_model_variants()
    
    # Explain attention
    print_attention_mechanism()
    
    # Explain preprocessing
    print_preprocessing_pipeline()
    
    print("\n" + "="*80)
    print("✅ Visualization Complete!")
    print("="*80 + "\n")
    
    print("For more details:")
    print("  - Check model.py for implementation")
    print("  - Check preprocessing.py for STFT details")
    print("  - Check README.md for full documentation")
    print()


if __name__ == "__main__":
    main()
