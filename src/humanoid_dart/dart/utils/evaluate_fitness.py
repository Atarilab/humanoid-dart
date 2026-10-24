import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def compute_laplacian(points):
    """
    Compute Laplacian coordinates for a set of points.
    points: (N, 3) array
    Returns: (N, 3) array of Laplacian coordinates
    """
    N = points.shape[0]
    try:
        tri = Delaunay(points)
    except Exception as e:
        print(f"Delaunay failed: {e}")
        return np.zeros_like(points)
        
    # Build adjacency
    neighbors = {i: set() for i in range(N)}
    for simplex in tri.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                u, v = simplex[i], simplex[j]
                neighbors[u].add(v)
                neighbors[v].add(u)
    
    laplacian_coords = np.zeros_like(points)
    for i in range(N):
        nbrs = list(neighbors[i])
        if len(nbrs) == 0:
            continue
        nbr_centroid = np.mean(points[nbrs], axis=0)
        laplacian_coords[i] = points[i] - nbr_centroid
        
    return laplacian_coords

def compute_laplacian_metrics(file_path, expert_file=None, env_idx=0):
    try:
        data = np.load(file_path)
        body_pos = data['body_pos_w'] # (T, Envs, N, 3)
        
        # Check if reference is available in the recorded file
        if 'ref_body_pos' in data:
            ref_body_pos = data['ref_body_pos'] # (T, Envs, N, 3)
            print("Found reference body positions in recorded file.")
        else:
            ref_body_pos = None
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    T, num_envs, N, _ = body_pos.shape
    
    if env_idx >= num_envs:
        print(f"Error: Environment index {env_idx} out of bounds.")
        return

    # Select environment
    points_sequence = body_pos[:, env_idx, :, :]
    
    # Handle expert data
    expert_sequence = None
    if expert_file:
        try:
            expert_data = np.load(expert_file)
            expert_full = expert_data['body_pos_w']
            
            # Check shape
            # Expert might be (T_exp, N_exp, 3) or (T_exp, Envs, N_exp, 3)
            if expert_full.ndim == 3:
                # (T, N, 3)
                expert_sequence = expert_full
            
            if expert_sequence is not None:
                # print(f"Loaded expert data with shape {expert_sequence.shape}")
                
                # Align lengths
                min_len = min(len(points_sequence), len(expert_sequence))
                points_sequence = points_sequence[:min_len]
                expert_sequence = expert_sequence[:min_len]
                T = min_len
            else:
                print("Error: Expert sequence could not be extracted.")
                return
            
        except Exception as e:
            print(f"Error loading expert file: {e}")
            return
    elif ref_body_pos is not None:
        # Use the recorded reference as expert
        expert_sequence = ref_body_pos[:, env_idx, :, :]
        print(f"Using recorded reference as expert data. Shape: {expert_sequence.shape}")
    
    # Filter specific body indices if requested
    target_indices = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]
    
    if expert_sequence is not None:
        # print(f"Filtering expert for body indices: {target_indices}")
        expert_sequence = expert_sequence[:, target_indices, :]
    
    rec_laplacians = []
    exp_laplacians = []
    l2_diffs = []
    
    # print(f"Computing metrics for {T} frames...")
    
    for t in range(T):
        # Recorded
        rec_points = points_sequence[t]
        rec_lap = compute_laplacian(rec_points)
        rec_laplacians.append(rec_lap)
        
        if expert_sequence is not None:
            exp_points = expert_sequence[t]
            # Ensure same number of points
            if len(exp_points) != len(rec_points):
                print(f"Frame {t}: Point count mismatch (Rec: {len(rec_points)}, Exp: {len(exp_points)}). Skipping diff.")
                l2_diffs.append(0.0)
                exp_laplacians.append(np.zeros_like(rec_lap)) # Placeholder
                continue
                
            exp_lap = compute_laplacian(exp_points)
            exp_laplacians.append(exp_lap)
            
            # Compute L2 norm of difference
            # || L_rec - L_exp ||_2 (sum over all points, or mean?)
            # Usually Frobenius norm for matrices or sum of vector norms
            # Let's do sum of L2 norms of vectors
            diff = rec_lap - exp_lap
            l2_diff = np.sum(np.linalg.norm(diff, axis=1))
            l2_diffs.append(l2_diff)
            
    # Convert to arrays
    rec_laplacians = np.array(rec_laplacians)
    if expert_sequence is not None:
        exp_laplacians = np.array(exp_laplacians)
        l2_diffs = np.array(l2_diffs)
        
        # Plot L2 diff
        plt.figure(figsize=(10, 6))
        plt.plot(l2_diffs)
        plt.title(f"Laplacian L2 Difference (Rec vs Expert) - Env {env_idx}")
        plt.xlabel("Frame")
        plt.ylabel("L2 Difference")
        plt.grid(True)
        plt.savefig("laplacian_diff.png")
        plt.close()
        # print("Saved plot to laplacian_diff.png")
        
        # Save metrics
        np.savez("metrics.npz", 
                 rec_laplacians=rec_laplacians, 
                 exp_laplacians=exp_laplacians, 
                 laplacian_l2_diff=l2_diffs)
    else:
        np.savez("metrics.npz", rec_laplacians=rec_laplacians)
        
    # print(f"Saved metrics to metrics.npz")

def compute_box_height_metric(file_path, env_idx=0, last_percent=0.4):
    try:
        data = np.load(file_path)
        # Check if box_pos is available
        if 'object_pos_w' in data:
            box_pos = data['object_pos_w'] # (T, Envs, 3)
        else:
            print("Error: 'object_pos_w' not found in recorded file.")
            return
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    T, num_envs, _ = box_pos.shape
    
    if env_idx >= num_envs:
        print(f"Error: Environment index {env_idx} out of bounds.")
        return

    # Select environment
    box_sequence = box_pos[:, env_idx, :]
    
    # Last X% of frames
    start_idx = int(T * (1.0 - last_percent))
    last_segment = box_sequence[start_idx:]
    
    # Average height (z-axis is index 2)
    avg_height = np.mean(last_segment[:, 2])
    
    percent_str = int(last_percent * 100)
    # print(f"Average box height in last {percent_str}% of frames ({T-start_idx} frames): {avg_height:.4f}")
    
    # Plot box height over time
    plt.figure(figsize=(10, 6))
    plt.plot(box_sequence[:, 2])
    plt.axvline(x=start_idx, color='r', linestyle='--', label=f'Last {percent_str}% Start')
    plt.axhline(y=avg_height, color='g', linestyle='--', label=f'Avg Height (Last {percent_str}%): {avg_height:.2f}')
    plt.title(f"Box Height over Time - Env {env_idx}")
    plt.xlabel("Frame")
    plt.ylabel("Height (Z)")
    plt.legend()
    plt.grid(True)
    output_plot = "box_height.png"
    plt.savefig(output_plot)
    plt.close()
    # print(f"Saved plot to {output_plot}")
    
    # Append to metrics.npz if it exists, or create new
    metric_key = f'avg_box_height_last_{percent_str}'
    try:
        metrics = dict(np.load("metrics.npz"))
        metrics[metric_key] = avg_height
        np.savez("metrics.npz", **metrics)
        # print(f"Updated metrics.npz with {metric_key}")
    except FileNotFoundError:
        kwargs = {metric_key: avg_height}
        np.savez("metrics.npz", **kwargs)
        print(f"Created metrics.npz with {metric_key}")

def compute_contact_metric(file_path, env_idx=0):
    try:
        data = np.load(file_path)
        if 'reward_contact_match' in data:
            contact_reward = data['reward_contact_match'] # (T, Envs)
        else:
            print("Error: 'reward_contact_match' not found in recorded file.")
            return
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    T, num_envs = contact_reward.shape
    
    if env_idx >= num_envs:
        print(f"Error: Environment index {env_idx} out of bounds.")
        return

    # Select environment
    contact_seq = contact_reward[:, env_idx]
    
    # Determine if contact was maintained
    # Assuming reward > 0 means contact.
    # We can compute the percentage of frames with non-zero reward
    contact_frames = np.sum(contact_seq > 1e-6)
    contact_percentage = (contact_frames / T) * 100.0
    
    # print(f"Contact maintained for {contact_frames}/{T} frames ({contact_percentage:.2f}%)")
    
    # Plot contact reward
    plt.figure(figsize=(10, 6))
    plt.plot(contact_seq)
    plt.title(f"Contact Reward over Time - Env {env_idx}")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    output_plot = "contact_reward.png"
    plt.savefig(output_plot)
    plt.close()
    # print(f"Saved plot to {output_plot}")
    
    # Update metrics.npz
    metric_key = 'contact_percentage'
    try:
        metrics = dict(np.load("metrics.npz"))
        metrics[metric_key] = contact_percentage
        np.savez("metrics.npz", **metrics)
        # print(f"Updated metrics.npz with {metric_key}")
    except FileNotFoundError:
        kwargs = {metric_key: contact_percentage}
        np.savez("metrics.npz", **kwargs)
        print(f"Created metrics.npz with {metric_key}")

def compute_reward_metrics(file_path, env_idx=0):
    try:
        data = np.load(file_path)
        metrics_to_extract = ["reward_motion_body_pos", "reward_motion_body_lin_vel"]
        results = {}
        
        for metric in metrics_to_extract:
            if metric in data:
                # Shape (T, Envs)
                val = data[metric][:, env_idx]
                mean_val = np.mean(val)
                results[f"mean_{metric}"] = mean_val
            else:
                # print(f"Warning: {metric} not found in {file_path}")
                results[f"mean_{metric}"] = 0.0
        
        # Update metrics.npz
        try:
            existing = dict(np.load("metrics.npz"))
            existing.update(results)
            np.savez("metrics.npz", **existing)
        except FileNotFoundError:
            np.savez("metrics.npz", **results)
            
    except Exception as e:
        print(f"Error in compute_reward_metrics: {e}")

def compute_fitness(env_idx=0):
    try:
        metrics = dict(np.load("metrics.npz"))
        
        # 1. Pose
        pose_score = float(metrics.get("mean_reward_motion_body_pos", 0.0))
        
        # 2. Velocity
        vel_score = float(metrics.get("mean_reward_motion_body_lin_vel", 0.0))
        
        # 3. Contact (percentage 0-100 -> 0-1)
        contact_pct = float(metrics.get("contact_percentage", 0.0))
        contact_score = contact_pct / 100.0
        
        # 4. Laplacian (L2 diff -> score)
        # Using exp(-diff)
        if "laplacian_l2_diff" in metrics:
            lap_diff = metrics["laplacian_l2_diff"]
            mean_lap_diff = np.mean(lap_diff)
            laplacian_score = np.exp(-mean_lap_diff)
        else:
            laplacian_score = 0.0
            
        fitness = pose_score * vel_score * contact_score * laplacian_score
        
        metrics["fitness_score"] = fitness
        metrics["laplacian_score"] = laplacian_score # Save intermediate
        
        np.savez("metrics.npz", **metrics)
        
    except Exception as e:
        print(f"Error computing fitness: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mesh energy based on Laplacian coordinates.")
    parser.add_argument("--file", type=str, default=None, help="Path to recorded positions npz.")
    parser.add_argument("--expert_file", type=str, default=None, help="Path to expert motion npz (optional).")
    parser.add_argument("--env", type=int, default=0, help="Environment index to analyze.")
    parser.add_argument("--last_percent", type=float, default=0.3, help="Fraction of last frames to compute box height (default: 0.3).")
    args = parser.parse_args()
    
    compute_laplacian_metrics(args.file, args.expert_file, args.env)
    compute_box_height_metric(args.file, args.env, args.last_percent)
    compute_contact_metric(args.file, args.env)
    compute_reward_metrics(args.file, args.env)
    compute_fitness(args.env)
