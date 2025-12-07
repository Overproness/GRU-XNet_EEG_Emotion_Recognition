"""
Dataset loaders for DEAP, GAMEEMO, and SEED-IV datasets
Handles loading, preprocessing, and batching of EEG data
"""

import numpy as np
import pickle
import os
from typing import Tuple, List, Dict, Optional
from pathlib import Path

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. SEED-IV loader will have limited functionality.")


class BaseDatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, data_dir: str, dataset_name: str):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.sampling_rate = None
        self.n_channels = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset and return (data, labels)"""
        raise NotImplementedError
    
    def get_subject_data(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load data for specific subject"""
        raise NotImplementedError
    
    def print_info(self):
        """Print dataset information"""
        print(f"\n{'='*80}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        print(f"Number of Channels: {self.n_channels}")
        print(f"{'='*80}\n")


class DEAPLoader(BaseDatasetLoader):
    """
    Loader for DEAP dataset
    - 32 subjects
    - 40 trials per subject (1 minute each)
    - 32 EEG channels + 8 peripheral
    - Sampling rate: 128 Hz (downsampled from 512 Hz)
    - Labels: Valence, Arousal, Dominance, Liking (1-9 scale)
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "DEAP")
        self.sampling_rate = 128
        self.n_channels = 32  # EEG channels only
        self.n_subjects = 32
        self.n_trials = 40
        self.trial_duration = 60  # seconds
        
    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for one subject
        
        Args:
            subject_id: Subject ID (1-32)
            
        Returns:
            data: shape (n_trials, n_channels, n_samples)
            labels: shape (n_trials, 4) - [valence, arousal, dominance, liking]
        """
        if subject_id < 1 or subject_id > 32:
            raise ValueError(f"Subject ID must be 1-32, got {subject_id}")
        
        # DEAP uses sXX.dat format
        filepath = self.data_dir / "data_preprocessed_python" / f"s{subject_id:02d}.dat"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load pickle file
        with open(filepath, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')
        
        # Extract data and labels
        # data shape: (40 trials, 40 channels, 8064 samples)
        # We take first 32 channels (EEG only)
        data = subject_data['data'][:, :32, :]  # (40, 32, 8064)
        
        # Labels: (40 trials, 4 labels) - valence, arousal, dominance, liking
        labels = subject_data['labels']  # (40, 4)
        
        return data, labels
    
    def load_all_subjects(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all subjects
        
        Returns:
            data: shape (n_total_trials, n_channels, n_samples)
            labels: shape (n_total_trials, 4)
            subject_ids: shape (n_total_trials,) - which subject each trial belongs to
        """
        all_data = []
        all_labels = []
        all_subject_ids = []
        
        print(f"Loading DEAP dataset...")
        for subject_id in range(1, self.n_subjects + 1):
            try:
                data, labels = self.load_subject(subject_id)
                all_data.append(data)
                all_labels.append(labels)
                all_subject_ids.extend([subject_id] * len(data))
                print(f"  Subject {subject_id:02d}: {data.shape[0]} trials loaded")
            except FileNotFoundError:
                print(f"  Subject {subject_id:02d}: File not found, skipping")
                continue
        
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)
        
        print(f"\nTotal loaded: {all_data.shape[0]} trials from {len(set(all_subject_ids))} subjects")
        return all_data, all_labels, all_subject_ids
    
    def create_binary_labels(self, labels: np.ndarray, dimension: str = 'valence', 
                            threshold: float = 5.0) -> np.ndarray:
        """
        Create binary labels from continuous ratings
        
        Args:
            labels: Original labels (n_trials, 4)
            dimension: 'valence', 'arousal', 'dominance', or 'liking'
            threshold: Threshold for binarization (default: 5.0)
            
        Returns:
            Binary labels (0 or 1)
        """
        dimension_map = {'valence': 0, 'arousal': 1, 'dominance': 2, 'liking': 3}
        idx = dimension_map[dimension]
        
        binary_labels = (labels[:, idx] >= threshold).astype(int)
        return binary_labels


class GAMEEMOLoader(BaseDatasetLoader):
    """
    Loader for GAMEEMO dataset
    - 28 subjects
    - 4 gaming sessions per subject (G1, G2, G3, G4)
    - 14 EEG channels (Emotiv EPOC)
    - Sampling rate: 128 Hz
    - Labels: SAM ratings (Valence, Arousal, Dominance) - in PDF, using default values
    """
    
    def __init__(self, data_dir: str, window_size: int = 128 * 5, overlap: float = 0.5):
        """
        Args:
            data_dir: Path to GAMEEMO dataset
            window_size: Window size in samples (default: 5 seconds at 128Hz = 640 samples)
            overlap: Overlap ratio (default: 0.5 for 50% overlap)
        """
        super().__init__(data_dir, "GAMEEMO")
        self.sampling_rate = 128
        self.n_channels = 14  # Emotiv EPOC channels
        self.n_subjects = 28
        self.n_games = 4  # G1, G2, G3, G4
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
        # Channel names
        self.channel_names = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 
                              'O1', 'O2', 'P7', 'P8', 'T7', 'T8']
    
    def _window_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Split continuous signal into windows
        
        Args:
            data: Continuous EEG data (n_channels, n_samples)
            
        Returns:
            Windowed data (n_windows, n_channels, window_size)
        """
        n_channels, n_samples = data.shape
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.step_size + 1
        
        windows = []
        for i in range(n_windows):
            start = i * self.step_size
            end = start + self.window_size
            if end <= n_samples:
                windows.append(data[:, start:end])
        
        return np.array(windows)
    
    def load_subject(self, subject_id: int, use_binary_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for one subject (all 4 games)
        
        Args:
            subject_id: Subject ID (1-28)
            use_binary_labels: If True, create binary labels (0=low arousal, 1=high arousal)
            
        Returns:
            data: Windowed EEG data (n_windows, n_channels, window_size)
            labels: Binary labels for each window (n_windows,)
        """
        if subject_id < 1 or subject_id > 28:
            raise ValueError(f"Subject ID must be 1-28, got {subject_id}")
        
        subject_folder = self.data_dir / f"(S{subject_id:02d})"
        
        if not subject_folder.exists():
            raise FileNotFoundError(f"Subject folder not found: {subject_folder}")
        
        csv_folder = subject_folder / "Preprocessed EEG Data" / ".csv format"
        if not csv_folder.exists():
            raise FileNotFoundError(f"CSV folder not found: {csv_folder}")
        
        all_windows = []
        all_labels = []
        
        # Load each game session
        for game_id in range(1, self.n_games + 1):
            csv_file = csv_folder / f"S{subject_id:02d}G{game_id}AllChannels.csv"
            
            if not csv_file.exists():
                print(f"  Warning: {csv_file.name} not found, skipping")
                continue
            
            try:
                # Load CSV
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                # Remove the unnamed column if present
                df = df[self.channel_names]
                
                # Convert to numpy (channels, samples)
                data = df.values.T
                
                # Window the signal
                windows = self._window_signal(data)
                
                # Create labels for this game
                # Since SAM ratings are in PDF, we'll use a simple heuristic:
                # G1, G2: lower arousal (label 0), G3, G4: higher arousal (label 1)
                # This is a simplification - ideally parse PDFs
                if use_binary_labels:
                    label = 0 if game_id <= 2 else 1
                else:
                    label = game_id - 1  # 0, 1, 2, 3
                
                labels = np.full(len(windows), label)
                
                all_windows.append(windows)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")
                continue
        
        if not all_windows:
            raise ValueError(f"No valid data loaded for subject {subject_id}")
        
        # Concatenate all windows
        data = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return data, labels
    
    def load_all_subjects(self, use_binary_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all subjects
        
        Args:
            use_binary_labels: If True, create binary labels
            
        Returns:
            data: All windowed data (n_total_windows, n_channels, window_size)
            labels: Labels for each window (n_total_windows,)
            subject_ids: Subject ID for each window (n_total_windows,)
        """
        all_data = []
        all_labels = []
        all_subject_ids = []
        
        print(f"Loading GAMEEMO dataset...")
        print(f"Window size: {self.window_size} samples ({self.window_size/self.sampling_rate:.1f}s)")
        print(f"Overlap: {self.overlap*100:.0f}%")
        
        for subject_id in range(1, self.n_subjects + 1):
            try:
                data, labels = self.load_subject(subject_id, use_binary_labels)
                
                all_data.append(data)
                all_labels.append(labels)
                all_subject_ids.extend([subject_id] * len(data))
                
                print(f"  Subject {subject_id:02d}: {len(data)} windows")
                
            except Exception as e:
                print(f"  Subject {subject_id:02d}: Failed - {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data loaded from any subject")
        
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)
        
        print(f"\nTotal loaded: {all_data.shape[0]} windows from {len(set(all_subject_ids))} subjects")
        print(f"Data shape: {all_data.shape}")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        return all_data, all_labels, all_subject_ids
    
    def create_binary_labels(self, labels: np.ndarray, threshold: int = 2) -> np.ndarray:
        """
        Create binary labels from game IDs
        
        Args:
            labels: Original labels (game IDs 0-3)
            threshold: Threshold for binarization (default: 2)
            
        Returns:
            Binary labels (0 or 1)
        """
        return (labels >= threshold).astype(int)


class SEEDIVLoader(BaseDatasetLoader):
    """
    Loader for SEED-IV dataset
    - 15 subjects (3 sessions each)
    - 24 trials per session
    - 62 EEG channels
    - Sampling rate: 200 Hz
    - Labels: 4 emotions (0=neutral, 1=sad, 2=fear, 3=happy)
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "SEED-IV")
        self.sampling_rate = 200
        self.n_channels = 62
        self.n_subjects = 15
        self.n_sessions = 3
        self.n_trials = 24
        
        # Standard trial length (will be determined from first session or fixed)
        self.target_trial_length = 28000  # ~140 seconds at 200 Hz
        
        # Session labels (24 trials each)
        self.session_labels = {
            1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
            3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
        }
        
    def load_subject_session(self, subject_id: int, session: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for one subject's session
        
        Args:
            subject_id: Subject ID (1-15)
            session: Session number (1-3)
            
        Returns:
            data: shape (n_trials, n_channels, n_samples)
            labels: shape (n_trials,)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for SEED-IV loading. Install with: pip install scipy")
        
        if subject_id < 1 or subject_id > 15:
            raise ValueError(f"Subject ID must be 1-15, got {subject_id}")
        if session < 1 or session > 3:
            raise ValueError(f"Session must be 1-3, got {session}")
        
        # Try different possible data structures
        possible_paths = [
            self.data_dir / "eeg_raw_data" / str(subject_id) / f"{subject_id}_{session}.mat",
            self.data_dir / "eeg_raw_data" / str(subject_id) / f"session{session}.mat",
            self.data_dir / "seed_iv" / f"s{subject_id:02d}_session{session}.mat",
            self.data_dir / f"{subject_id}_{session}_eeg.mat",
        ]
        
        # Also try glob patterns
        search_folders = [
            self.data_dir / "eeg_raw_data" / str(subject_id),
            self.data_dir / "eeg_raw_data",
            self.data_dir / "seed_iv",
            self.data_dir,
        ]
        
        session_file = None
        for folder in search_folders:
            if folder.exists():
                # Try different naming patterns
                patterns = [
                    f"*{subject_id}*{session}*.mat",
                    f"*s{subject_id:02d}*{session}*.mat",
                    f"*session{session}*.mat",
                ]
                for pattern in patterns:
                    files = list(folder.glob(pattern))
                    if files:
                        session_file = files[0]
                        break
            if session_file:
                break
        
        # Try direct paths
        if session_file is None:
            for path in possible_paths:
                if path.exists():
                    session_file = path
                    break
        
        if session_file is None:
            raise FileNotFoundError(
                f"Could not find data file for subject {subject_id}, session {session}.\n"
                f"Searched in: {search_folders}\n"
                f"Expected patterns: *{subject_id}*{session}*.mat"
            )
        
        # Load .mat file
        mat_data = loadmat(str(session_file))
        
        # Extract EEG data - SEED-IV typically stores trials as separate variables
        # Common formats: 'de_LDS1', 'de_LDS2', etc. or 'eeg_trial1', 'eeg_trial2', etc.
        trial_data = []
        
        # Try to find trial variables
        trial_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        # Check if data is in trial-based format
        if any('trial' in k.lower() or 'eeg' in k.lower() for k in trial_keys):
            # Sort keys to ensure correct order
            trial_keys = sorted([k for k in trial_keys if 'trial' in k.lower() or 'eeg' in k.lower()])
            for key in trial_keys[:24]:  # Maximum 24 trials
                trial = mat_data[key]
                if isinstance(trial, np.ndarray):
                    trial_data.append(trial)
        
        # Alternative: single array with all trials
        elif 'data' in mat_data:
            data_array = mat_data['data']
            if data_array.ndim == 3:  # (n_trials, n_channels, n_samples)
                trial_data = [data_array[i] for i in range(min(24, data_array.shape[0]))]
        
        # Alternative: check for common SEED-IV variable names
        else:
            for key in trial_keys:
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.ndim >= 2:
                    if data.shape[0] == 62 or data.shape[1] == 62:  # 62 channels
                        trial_data.append(data)
        
        if not trial_data:
            raise ValueError(
                f"Could not parse data from {session_file}.\n"
                f"Available keys: {trial_keys}\n"
                f"Please check the .mat file structure."
            )
        
        # Ensure consistent shape (n_channels, n_samples)
        # Handle variable-length trials using fixed target length
        standardized_trials = []
        
        for trial in trial_data:
            # Transpose if channels are in wrong dimension
            if trial.shape[0] != 62 and trial.shape[1] == 62:
                trial = trial.T
            
            n_channels, n_samples = trial.shape
            
            # Pad or crop to target length
            if n_samples < self.target_trial_length:
                # Pad with zeros
                padding = np.zeros((n_channels, self.target_trial_length - n_samples))
                trial = np.concatenate([trial, padding], axis=1)
            elif n_samples > self.target_trial_length:
                # Crop from center
                start_idx = (n_samples - self.target_trial_length) // 2
                trial = trial[:, start_idx:start_idx + self.target_trial_length]
            
            standardized_trials.append(trial)
        
        data = np.array(standardized_trials)  # (n_trials, n_channels, n_samples)
        
        # Get labels for this session
        labels = np.array(self.session_labels[session][:len(data)])
        
        return data, labels
    
    def load_all_subjects(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all subjects and sessions
        
        Returns:
            data: shape (n_total_trials, n_channels, n_samples)
            labels: shape (n_total_trials,)
            subject_ids: shape (n_total_trials,) - which subject each trial belongs to
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for SEED-IV loading. Install with: pip install scipy")
        
        all_data = []
        all_labels = []
        all_subject_ids = []
        
        print(f"Loading SEED-IV dataset...")
        for subject_id in range(1, self.n_subjects + 1):
            for session in range(1, self.n_sessions + 1):
                try:
                    data, labels = self.load_subject_session(subject_id, session)
                    
                    all_data.append(data)
                    all_labels.append(labels)
                    all_subject_ids.extend([subject_id] * len(data))
                    
                    print(f"  Loaded subject {subject_id}, session {session}: {data.shape[0]} trials")
                    
                except FileNotFoundError as e:
                    print(f"  Warning: Could not load subject {subject_id}, session {session}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error loading subject {subject_id}, session {session}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data could be loaded. Please check the dataset path and structure.")
        
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)
        
        print(f"\nTotal loaded: {all_data.shape[0]} trials from {len(set(all_subject_ids))} subjects")
        return all_data, all_labels, all_subject_ids


def get_loader(dataset_name: str, data_dir: str) -> BaseDatasetLoader:
    """
    Factory function to get appropriate dataset loader
    
    Args:
        dataset_name: 'DEAP', 'GAMEEMO', or 'SEED-IV'
        data_dir: Path to dataset directory
        
    Returns:
        Dataset loader instance
    """
    dataset_name = dataset_name.upper()
    
    if dataset_name == 'DEAP':
        return DEAPLoader(data_dir)
    elif dataset_name == 'GAMEEMO':
        return GAMEEMOLoader(data_dir)
    elif dataset_name in ['SEED-IV', 'SEEDIV', 'SEED_IV']:
        return SEEDIVLoader(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'DEAP', 'GAMEEMO', 'SEED-IV'")


if __name__ == "__main__":
    # Test loaders
    import sys
    
    base_dir = r""
    
    # Test DEAP loader
    print("Testing DEAP Loader")
    print("=" * 80)
    try:
        deap_loader = DEAPLoader(os.path.join(base_dir, "deap"))
        deap_loader.print_info()
        
        # Load one subject
        data, labels = deap_loader.load_subject(1)
        print(f"Subject 1 data shape: {data.shape}")
        print(f"Subject 1 labels shape: {labels.shape}")
        print(f"Sample labels: {labels[0]}")
        
        # Create binary labels
        binary_valence = deap_loader.create_binary_labels(labels, 'valence')
        print(f"Binary valence labels: {binary_valence}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test GAMEEMO loader
    print("\n\nTesting GAMEEMO Loader")
    print("=" * 80)
    try:
        gameemo_loader = GAMEEMOLoader(os.path.join(base_dir, "GAMEEMO"))
        gameemo_loader.print_info()
    except Exception as e:
        print(f"Error: {e}")
    
    # Test SEED-IV loader
    print("\n\nTesting SEED-IV Loader")
    print("=" * 80)
    try:
        seediv_loader = SEEDIVLoader(os.path.join(base_dir, "sead-4"))
        seediv_loader.print_info()
        print(f"Session 1 labels: {seediv_loader.session_labels[1]}")
    except Exception as e:
        print(f"Error: {e}")
