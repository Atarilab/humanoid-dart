import numpy as np
from pathlib import Path
import tyro
import matplotlib.pyplot as plt
import os
import math
from typing import Literal, Optional
from scipy.fftpack import dct, idct
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}

class DCTTrajectoryAugmenter:
    """
    Class to handle DCT-based trajectory augmentation.
    """
    def __init__(
        self,
        K: int = 20,
        scale: float = 0.3,
        sampler: Literal["gaussian", "sobol"] = "sobol",
        smoothing: Literal["savitzky_golay", "gaussian", None] = None,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 3
    ):
        self.K = K
        self.scale = scale
        self.sampler = sampler
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        
        # Define DOF range to perturb (12 to 28)
        self.dof_slice = slice(15, 29)

    def compute_dct_coefficients(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute K DCT coefficients for a trajectory.
        
        Args:
            trajectory: Shape (T, DOF) - time series trajectory
        
        Returns:
            coefficients: Shape (K, DOF) - DCT coefficients
        """
        T, DOF = trajectory.shape
        coefficients = np.zeros((self.K-1, DOF))
        
        for dof_idx in range(DOF):
            # Compute full DCT
            full_coeffs = dct(trajectory[:, dof_idx], type=2, norm='ortho')
            # Keep top K coefficients
            coefficients[:, dof_idx] = full_coeffs[1:self.K]
        
        return coefficients

    def sample_around_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Sample perturbed DCT coefficients around the expert coefficients.
        
        Args:
            coefficients: Shape (K, DOF) - expert DCT coefficients
        
        Returns:
            perturbed_coeffs: Shape (K, DOF) - sampled coefficients
        """
        K, DOF = coefficients.shape
        
        # Compute magnitude-based scales
        coeff_magnitudes = np.abs(coefficients)
        # coeff_magnitudes[coeff_magnitudes == 0] = 1.0  # Avoid zero scaling
        scales = self.scale * coeff_magnitudes
        
        if self.sampler == "gaussian":
            # Standard Gaussian noise scaled by coefficient magnitude
            noise = np.random.normal(0, 1, size=(K, DOF))
            
            perturbed_coeffs = coefficients.copy()
            perturbed_coeffs[:, self.dof_slice] += scales[:, self.dof_slice] * noise[:, self.dof_slice]
        
        elif self.sampler == "sobol":
            try:
                from scipy.stats import qmc
                # Sobol sampler: generates low-discrepancy sequence
                sampler_obj = qmc.Sobol(d=K * DOF, scramble=True)
                sobol_samples = sampler_obj.random(n=1)[0]  # Shape (K*DOF,)
                # Transform from [0,1] to [-1, 1]
                sobol_normalized = 2 * sobol_samples - 1
                # Reshape and scale
                noise = sobol_normalized.reshape(K, DOF)
                
                perturbed_coeffs = coefficients.copy()
                perturbed_coeffs[:, self.dof_slice] += scales[:, self.dof_slice] * noise[:, self.dof_slice]
            except ImportError:
                print("Warning: scipy.stats.qmc not available. Falling back to Gaussian.")
                noise = np.random.normal(0, 1, size=(K, DOF))
                
                perturbed_coeffs = coefficients.copy()
                perturbed_coeffs[:, self.dof_slice] += scales[:, self.dof_slice] * noise[:, self.dof_slice]
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        
        return perturbed_coeffs

    def reconstruct_trajectory(self, coefficients: np.ndarray, target_length: int) -> np.ndarray:
        """
        Reconstruct trajectory from DCT coefficients via inverse DCT.
        
        Args:
            coefficients: Shape (K, DOF) - DCT coefficients
            target_length: Target trajectory length T
        
        Returns:
            trajectory: Shape (T, DOF) - reconstructed trajectory
        """
        K, DOF = coefficients.shape
        trajectory = np.zeros((target_length, DOF))
        
        for dof_idx in range(DOF):
            # Zero-pad coefficients to target length
            padded_coeffs = np.zeros(target_length)
            padded_coeffs[:K] = coefficients[:, dof_idx]
            
            # Inverse DCT
            reconstructed = idct(padded_coeffs, type=2, norm='ortho')
            trajectory[:, dof_idx] = reconstructed
        
        # Apply smoothing to kill high-frequency artifacts
        if self.smoothing == "savitzky_golay":
            # Ensure window size is appropriate
            window = min(self.smoothing_window, target_length // 2 * 2 - 1)
            window = max(3, window)
            if window % 2 == 0:
                window -= 1
            
            for dof_idx in range(DOF):
                trajectory[:, dof_idx] = savgol_filter(
                    trajectory[:, dof_idx],
                    window_length=window,
                    polyorder=self.smoothing_polyorder
                )
        
        elif self.smoothing == "gaussian":
            # Gaussian blur via convolution
            sigma = 1.0
            window_size = int(4 * sigma + 1)
            if window_size % 2 == 0:
                window_size += 1
            x = np.arange(-window_size // 2, window_size // 2 + 1)
            kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel /= kernel.sum()
            
            for dof_idx in range(DOF):
                trajectory[:, dof_idx] = np.convolve(
                    trajectory[:, dof_idx],
                    kernel,
                    mode='same'
                )
        
        return trajectory

    def augment(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply DCT-based sampling to trajectory data.
        
        Args:
            data: Shape (T, DOF) - input trajectory
            mask: Shape (T,) or (T, DOF) - binary mask (1=augment, 0=keep)
        
        Returns:
            augmented_data: Shape (T, DOF) - augmented trajectory
        """
        T, DOF = data.shape
        
        # Compute DCT coefficients from expert trajectory
        coefficients = self.compute_dct_coefficients(data)
        
        # Reconstruct the "smoothed" expert trajectory (low-frequency component)
        smoothed_expert = self.reconstruct_trajectory(coefficients, T)
        
        # Calculate the high-frequency residuals (details that were lost)
        residuals = data - smoothed_expert

        # Sample perturbed coefficients
        perturbed_coeffs = self.sample_around_coefficients(coefficients)
        
        # Reconstruct trajectory from perturbed coefficients
        smoothed_augmented = self.reconstruct_trajectory(perturbed_coeffs, T)
        
        # Add residuals back to preserve local features
        augmented_data = smoothed_augmented + residuals
        
        if mask is not None:
            # Ensure mask is broadcastable
            if mask.ndim == 1:
                mask = mask[:, None]
            
            # Cast to float for blending
            mask = mask.astype(data.dtype)
            
            # Apply mask: 1.0 -> augmented, 0.0 -> original
            augmented_data = data * (1.0 - mask) + augmented_data * mask

        return augmented_data

class GaussianTrajectoryAugmenter:
    """
    Class to handle pure Gaussian trajectory augmentation.
    Adds Gaussian noise directly to the trajectory.
    """
    def __init__(
        self,
        scale: float = 0.05,
    ):
        self.scale = scale

    def augment(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Gaussian noise to trajectory data.
        
        Args:
            data: Shape (T, DOF) - input trajectory
            mask: Shape (T,) or (T, DOF) - binary mask (1=augment, 0=keep)
        
        Returns:
            augmented_data: Shape (T, DOF) - augmented trajectory
        """
        # Generate Gaussian noise
        noise = np.random.normal(0, self.scale, size=data.shape)
        
        # Add noise to data
        augmented_data = data + noise
        
        if mask is not None:
            # Ensure mask is broadcastable
            if mask.ndim == 1:
                mask = mask[:, None]
            
            # Cast to float for blending
            mask = mask.astype(data.dtype)
            
            # Apply mask: 1.0 -> augmented, 0.0 -> original
            augmented_data = data * (1.0 - mask) + augmented_data * mask

        return augmented_data

class SmoothNoiseAugmenter:
    """
    Adds smooth (low-frequency) noise to the trajectory.
    This avoids the 'jittery' look of pure Gaussian noise while still exploring.
    """
    def __init__(self, scale: float = 0.05, sigma: float = 10.0):
        self.scale = scale
        self.sigma = sigma

    def augment(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply smooth noise to trajectory data.
        
        Args:
            data: Shape (T, DOF) - input trajectory
            mask: Shape (T,) or (T, DOF) - binary mask (1=augment, 0=keep)
        """
        T, DOF = data.shape
        # Generate white noise
        noise = np.random.normal(0, 1.0, size=(T, DOF))
        
        # Smooth the noise along the time dimension
        # sigma controls the smoothness (higher = smoother/lower freq)
        smooth_noise = gaussian_filter1d(noise, sigma=self.sigma, axis=0)
        
        # Re-scale to match desired amplitude
        # Smoothing reduces variance, so we normalize to unit std then scale
        current_std = np.std(smooth_noise)
        if current_std > 1e-9:
            smooth_noise = smooth_noise / current_std * self.scale
        else:
            smooth_noise = noise * self.scale # Fallback if smoothing killed everything
            
        augmented_data = data + smooth_noise
        
        if mask is not None:
            # Ensure mask is broadcastable
            if mask.ndim == 1:
                mask = mask[:, None]
            
            # Cast to float for blending
            mask = mask.astype(data.dtype)
            
            # Apply mask: 1.0 -> augmented, 0.0 -> original
            augmented_data = data * (1.0 - mask) + augmented_data * mask

        return augmented_data

class SinusoidTrajectoryAugmenter:
    """
    Injects sinusoidal noise into specific joints.
    """
    def __init__(
        self,
        dt: float = 0.02 
    ):
        self.dt = dt
        # Configuration
        self.joints = list([15, 18, 22, 25]) # Default to arm joints
        self.freq_range = (0, 1.0)
        self.amplitude = 0.2
        self.phase_range = None  # Full [0, 2pi] range

    def augment(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply sinusoidal noise to trajectory data.
        
        Args:
            data: Shape (T, DOF) - input trajectory
            mask: Shape (T,) or (T, DOF) - binary mask (1=augment, 0=keep)
        """
        T, DOF = data.shape
        augmented_data = data.copy()
        
        time = np.arange(T) * self.dt
        
        joints_to_augment = self.joints if self.joints is not None else range(DOF)

        for joint_idx in joints_to_augment:
            if joint_idx >= DOF:
                continue
                
            # Sample frequency
            freq = np.random.uniform(self.freq_range[0], self.freq_range[1])
            
            # Sample phase
            if self.phase_range:
                phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
            else:
                phase = np.random.uniform(0, 2 * np.pi)
                
            # Generate sinusoid
            noise = self.amplitude * np.sin(2 * np.pi * freq * time + phase)
            
            augmented_data[:, joint_idx] += noise
            
        if mask is not None:
            # Ensure mask is broadcastable
            if mask.ndim == 1:
                mask = mask[:, None]
            
            # Cast to float for blending
            mask = mask.astype(data.dtype)
            
            # Apply mask: 1.0 -> augmented, 0.0 -> original
            augmented_data = data * (1.0 - mask) + augmented_data * mask

        return augmented_data

def plot_before_after(original_data, augmented_data, output_dir):
    """
    Plot before/after trajectories for joint_pos and joint_vel.
    Uses a grid layout for better readability.
    """

    keys_to_plot = ["joint_pos", "joint_vel"]
    os.makedirs(output_dir, exist_ok=True)

    for key in keys_to_plot:
        if key not in original_data:
            print(f"Key '{key}' not found. Skipping.")
            continue

        orig = original_data[key]
        aug  = augmented_data[key]

        if orig.ndim != 2:
            print(f"Skipping {key}: expected (T, D), got {orig.shape}")
            continue

        T, D = orig.shape

        # ---- GRID SETUP ----
        cols = math.ceil(math.sqrt(D))
        rows = math.ceil(D / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
        axes = axes.flatten()

        fig.suptitle(f"Original vs Augmented: {key}", fontsize=16)

        # ---- PLOT EACH DIMENSION ----
        for i in range(D):
            ax = axes[i]
            ax.plot(orig[:, i], label="Original", alpha=0.8)
            ax.plot(aug[:, i], label="Augmented", alpha=0.8)
            ax.set_title(f"Joint {i}", fontsize=10)
            ax.grid(True)

        # Hide unused subplots if grid > D
        for j in range(D, len(axes)):
            axes[j].axis("off")

        # Add legend once
        axes[0].legend()

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        save_path = os.path.join(output_dir, f"{key}.png")
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(output_dir, f"{key}_{counter}.png")
            counter += 1

        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Saved grid plot → {save_path}")

def augment_trajectory_data(
    traj_data: dict,
    method: Literal["dct", "gaussian", "smooth", "sinusoid"] = "dct",
    mask_start_idx: int = 250
) -> dict:
    """
    Augment trajectory data dictionary in-memory.
    
    Args:
        traj_data: Dictionary containing trajectory arrays
        method: Augmentation method
        mask_start_idx: Time step index to start augmentation from
        
    Returns:
        augmented_data: Dictionary with augmented arrays
    """
    if method == "dct":
        augmenter = DCTTrajectoryAugmenter(sampler='gaussian')
    elif method == "gaussian":
        augmenter = GaussianTrajectoryAugmenter(scale=0.01)
    elif method == "smooth":
        augmenter = SmoothNoiseAugmenter(scale=0.05, sigma=10.0)
    elif method == "sinusoid":
        fps = float(traj_data.get("fps", 50.0))
        dt = 1.0 / fps
        augmenter = SinusoidTrajectoryAugmenter(dt=dt)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    # Mask definition
    mask = None
    # if "joint_pos" in traj_data:
    #     T = traj_data["joint_pos"].shape[0]
    #     mask = np.zeros(T, dtype=np.float32)
        
    #     # Define mask regions here (1.0 = augment, 0.0 = keep original)
    #     if mask_start_idx < T:
    #         mask[:] = 1.0  
        
    # Augmentation: 
    augmented_data = traj_data.copy()
    if "joint_pos" in augmented_data:
        augmented_data["joint_pos"] = augmenter.augment(augmented_data["joint_pos"], mask=mask)
            
    return augmented_data

def main(
        input_npz: Path,
        output_npz: Path,
        method: Literal["dct", "gaussian", "smooth", "sinusoid"] = "dct",
        plot: bool = True,
    ):
        """
        Augment trajectory data using DCT-based, Gaussian, Smooth noise, or Sinusoid injection.
        
        Args:
            input_npz: Path to input NPZ file
            output_npz: Path to output NPZ file
            method: Augmentation method ('dct', 'gaussian', 'smooth', 'sinusoid')
            plot: Whether to plot before/after trajectories
        """
        traj_data = load_npz(input_npz)
        
        augmented_data = augment_trajectory_data(traj_data, method=method)

        # Save Augmented Trajectory
        np.savez(output_npz, **augmented_data)
        # print(f"\nAugmented trajectory saved to {output_npz}")
        
        if plot:
            plot_before_after(original_data=traj_data, augmented_data=augmented_data, output_dir=output_npz.parent)

    
if __name__ == "__main__":
     tyro.cli(main)
    

