#!/usr/bin/env python3
"""
Spectral Structure Explorer for Jacobi NPZ files
Explores FFT, wavelet, and spectral graph interpretations of the stored data.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft, fft2, fftfreq, fftshift
from scipy.signal import cwt, morlet2
from scipy.linalg import svd, eigvals
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
import glob, os, sys, re

def load_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        arr = data[key]
        result[key] = arr.item() if arr.ndim == 0 else arr
    data.close()
    return result

def explore_spectral_structures(filepath, output_dir="spectral_analysis"):
    """Run all spectral analyses on a single npz file."""
    os.makedirs(output_dir, exist_ok=True)
    data = load_npz(filepath)
    basename = os.path.basename(filepath).replace('.npz', '')
    
    print(f"\n{'='*60}")
    print(f"Exploring: {basename}")
    print(f"Keys: {list(data.keys())}")
    print(f"{'='*60}")
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════════
    # A. FFT of the Jacobian matrix
    # ═══════════════════════════════════════════════════════════════
    if 'jacobian' in data:
        J = data['jacobian']
        print(f"\nJacobian shape: {J.shape}")
        
        # 2D FFT of the Jacobian — reveals periodic structure in the
        # linear map between layers
        J_fft = fft2(J)
        J_power = np.abs(J_fft)**2
        J_phase = np.angle(J_fft)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0a0a1a')
        
        # Power spectrum
        ax = axes[0, 0]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(fftshift(J_power)), cmap='inferno', aspect='auto')
        ax.set_title('Jacobian 2D FFT Power (log)', color='#c0d8e8')
        plt.colorbar(im, ax=ax)
        
        # Phase spectrum
        ax = axes[0, 1]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(fftshift(J_phase), cmap='hsv', aspect='auto')
        ax.set_title('Jacobian 2D FFT Phase', color='#c0d8e8')
        plt.colorbar(im, ax=ax)
        
        # 1D FFT along rows (token dimension)
        row_ffts = np.abs(fft(J, axis=1))**2
        ax = axes[0, 2]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(row_ffts[:, :J.shape[1]//2]), 
                       cmap='magma', aspect='auto')
        ax.set_title('Row-wise FFT Power (frequency modes per dim)', color='#c0d8e8')
        plt.colorbar(im, ax=ax)
        
        # Eigenvalue spectrum of J in polar form
        eigs = eigvals(J)
        ax = axes[1, 0]
        ax.set_facecolor('#0d1117')
        ax.scatter(eigs.real, eigs.imag, c=np.abs(eigs), cmap='plasma', 
                   s=20, alpha=0.8)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), '--', color='#555', lw=0.5)
        ax.set_aspect('equal')
        ax.set_title('Eigenvalue Spectrum (complex plane)', color='#c0d8e8')
        
        # FFT of eigenvalue magnitudes (sorted) — reveals spectral gaps
        eig_mags = np.sort(np.abs(eigs))[::-1]
        eig_fft = np.abs(fft(eig_mags))**2
        ax = axes[1, 1]
        ax.set_facecolor('#0d1117')
        ax.plot(eig_fft[1:len(eig_fft)//2], color='#44aaff', lw=1.5)
        ax.set_title('FFT of Eigenvalue Magnitude Sequence', color='#c0d8e8')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        
        # Singular value gaps — where are the "jumps"?
        U, S, Vh = svd(J)
        sv_diffs = np.diff(S)
        ax = axes[1, 2]
        ax.set_facecolor('#0d1117')
        ax.bar(range(len(sv_diffs)), np.abs(sv_diffs), color='#ff44aa', alpha=0.8)
        ax.set_title('Singular Value Gaps (|Δσ_i|)', color='#c0d8e8')
        ax.set_xlabel('Index')
        
        # Detect spectral clusters
        gap_threshold = np.mean(np.abs(sv_diffs)) + 2*np.std(np.abs(sv_diffs))
        big_gaps = np.where(np.abs(sv_diffs) > gap_threshold)[0]
        if len(big_gaps) > 0:
            print(f"  Spectral gaps at indices: {big_gaps}")
            print(f"  → This suggests {len(big_gaps)+1} distinct 'modes' in the Jacobian")
            for g in big_gaps:
                ax.axvline(g, color='#ffcc44', lw=1, ls='--', alpha=0.7)
        
        results['n_spectral_modes'] = len(big_gaps) + 1
        results['spectral_gap_indices'] = big_gaps.tolist()
        
        plt.suptitle(f'Spectral Analysis — {basename}', color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_spectral.png', dpi=150, 
                    facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════
    # B. FFT of per-token sequences (volume, rotation, shear)
    # ═══════════════════════════════════════════════════════════════
    token_signals = {}
    for key in ['per_token_volume', 'per_token_rotation', 'per_token_shear_mag']:
        if key in data:
            token_signals[key] = data[key]
    
    if token_signals:
        fig, axes = plt.subplots(len(token_signals), 3, 
                                 figsize=(18, 5*len(token_signals)), facecolor='#0a0a1a')
        if len(token_signals) == 1:
            axes = axes[np.newaxis, :]
        
        for idx, (key, signal) in enumerate(token_signals.items()):
            # Raw signal
            ax = axes[idx, 0]
            ax.set_facecolor('#0d1117')
            ax.plot(signal, color='#44aaff', lw=1)
            ax.set_title(f'{key} (raw)', color='#c0d8e8', fontsize=9)
            
            # FFT power spectrum
            N = len(signal)
            freqs = fftfreq(N)[:N//2]
            fft_vals = fft(signal)
            power = np.abs(fft_vals[:N//2])**2
            
            ax = axes[idx, 1]
            ax.set_facecolor('#0d1117')
            ax.plot(freqs[1:], power[1:], color='#ff44aa', lw=1)
            ax.set_title(f'{key} FFT Power', color='#c0d8e8', fontsize=9)
            ax.set_xlabel('Frequency (cycles/token)')
            
            # Detect dominant frequencies
            if len(power) > 3:
                peak_freqs = np.argsort(power[1:])[::-1][:5] + 1
                dominant_periods = 1.0 / (freqs[peak_freqs] + 1e-10)
                print(f"  {key}: dominant periods = {dominant_periods[:3]}")
                results[f'{key}_dominant_periods'] = dominant_periods[:3].tolist()
            
            # Continuous wavelet transform (multi-scale analysis)
            if len(signal) > 10:
                widths = np.arange(1, min(N//2, 30))
                cwt_matrix = cwt(signal, morlet2, widths)
                
                ax = axes[idx, 2]
                ax.set_facecolor('#0d1117')
                im = ax.imshow(np.abs(cwt_matrix), cmap='magma', aspect='auto',
                              extent=[0, N, widths[-1], widths[0]])
                ax.set_title(f'{key} Wavelet Scalogram', color='#c0d8e8', fontsize=9)
                ax.set_xlabel('Token position')
                ax.set_ylabel('Scale')
        
        plt.suptitle(f'Per-Token Spectral Analysis — {basename}', 
                     color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_token_spectral.png', dpi=150,
                    facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════
    # C. Graph Laplacian spectrum of the point cloud
    # ═══════════════════════════════════════════════════════════════
    for cloud_key in ['h_out_2d', 'h_in_2d']:
        if cloud_key not in data:
            continue
        
        points = data[cloud_key]
        if points.shape[0] < 5:
            continue
        
        # Build k-NN graph
        dists = squareform(pdist(points))
        k = min(10, points.shape[0] - 1)
        
        # Adjacency: connect each point to its k nearest neighbors
        A = np.zeros_like(dists)
        for i in range(len(dists)):
            neighbors = np.argsort(dists[i])[1:k+1]
            A[i, neighbors] = 1
            A[neighbors, i] = 1  # symmetric
        
        # Graph Laplacian
        L = laplacian(A, normed=True)
        lap_eigs = np.sort(np.real(eigvals(L.toarray() if hasattr(L, 'toarray') else L)))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0a1a')
        
        # Laplacian eigenvalues
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        ax.plot(lap_eigs, 'o-', color='#44ffaa', markersize=3, lw=1)
        ax.set_title(f'Graph Laplacian Spectrum ({cloud_key})', color='#c0d8e8')
        ax.set_xlabel('Index')
        ax.set_ylabel('λ')
        
        # Count near-zero eigenvalues (= number of connected components)
        n_components = np.sum(lap_eigs < 1e-6)
        print(f"  {cloud_key} graph: {n_components} connected components (from Laplacian)")
        
        # Spectral gap — indicates how "clustered" the data is
        if len(lap_eigs) > n_components:
            spectral_gap = lap_eigs[n_components]  # first non-zero eigenvalue
            print(f"  Spectral gap: {spectral_gap:.4f} (larger = more separated clusters)")
            results[f'{cloud_key}_spectral_gap'] = spectral_gap
            results[f'{cloud_key}_n_components'] = n_components
        
        # Fiedler vector (2nd smallest eigenvector) — natural partition
        from scipy.sparse.linalg import eigsh
        try:
            L_dense = L.toarray() if hasattr(L, 'toarray') else L
            _, eigvecs = np.linalg.eigh(L_dense)
            fiedler = eigvecs[:, n_components]  # first non-trivial eigenvector
            
            ax = axes[1]
            ax.set_facecolor('#0d1117')
            scatter = ax.scatter(points[:, 0], points[:, 1], c=fiedler, 
                               cmap='RdBu_r', s=20, alpha=0.8)
            ax.set_title('Fiedler Vector Coloring (natural partition)', color='#c0d8e8')
            plt.colorbar(scatter, ax=ax)
        except Exception:
            pass
        
        # FFT of the distance matrix rows (periodic structure in neighborhoods)
        dist_fft = np.abs(fft(dists, axis=1))**2
        mean_dist_power = dist_fft.mean(axis=0)
        
        ax = axes[2]
        ax.set_facecolor('#0d1117')
        ax.plot(mean_dist_power[1:len(mean_dist_power)//2], color='#ffaa44', lw=1.5)
        ax.set_title('Mean FFT Power of Distance Rows', color='#c0d8e8')
        ax.set_xlabel('Frequency mode')
        
        plt.suptitle(f'Graph Spectral Analysis — {basename}', color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_{cloud_key}_graph_spectral.png', 
                    dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════
    # D. Grid deformation as a vector field — curl/divergence spectra
    # ═══════════════════════════════════════════════════════════════
    if all(k in data for k in ['grid_x_in', 'grid_y_in', 'grid_x_out', 'grid_y_out']):
        gx_in, gy_in = data['grid_x_in'], data['grid_y_in']
        gx_out, gy_out = data['grid_x_out'], data['grid_y_out']
        
        # Displacement field
        dx = gx_out - gx_in
        dy = gy_out - gy_in
        
        # 2D FFT of displacement components
        dx_fft = fft2(dx)
        dy_fft = fft2(dy)
        
        # Helmholtz decomposition in Fourier space:
        # Any vector field F = ∇φ + ∇×ψ (irrotational + solenoidal)
        # In Fourier space: F_hat = ik*φ_hat + ik×ψ_hat
        N = dx.shape[0]
        kx = fftfreq(N)[:, np.newaxis]
        ky = fftfreq(N)[np.newaxis, :]
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1  # avoid division by zero
        
        # Divergence in Fourier space: div(F) = ik_x*F_x + ik_y*F_y
        div_fft = 1j * kx * dx_fft + 1j * ky * dy_fft
        
        # Curl in Fourier space (2D): curl(F) = ik_x*F_y - ik_y*F_x
        curl_fft = 1j * kx * dy_fft - 1j * ky * dx_fft
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0a0a1a')
        
        ax = axes[0, 0]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(np.abs(fftshift(dx_fft))**2), cmap='inferno', aspect='auto')
        ax.set_title('dx FFT Power', color='#c0d8e8')
        
        ax = axes[0, 1]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(np.abs(fftshift(dy_fft))**2), cmap='inferno', aspect='auto')
        ax.set_title('dy FFT Power', color='#c0d8e8')
        
        ax = axes[0, 2]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(np.abs(fftshift(div_fft))**2), cmap='RdBu_r', aspect='auto')
        ax.set_title('Divergence Power Spectrum', color='#c0d8e8')
        
        ax = axes[1, 0]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(np.log1p(np.abs(fftshift(curl_fft))**2), cmap='PuOr', aspect='auto')
        ax.set_title('Curl Power Spectrum', color='#c0d8e8')
        
        # Irrotational vs solenoidal energy ratio
        irrot_energy = np.sum(np.abs(div_fft)**2)
        sol_energy = np.sum(np.abs(curl_fft)**2)
        total_energy = irrot_energy + sol_energy
        
        print(f"  Helmholtz decomposition:")
        print(f"    Irrotational (∇φ) energy: {irrot_energy/total_energy:.1%}")
        print(f"    Solenoidal (∇×ψ) energy: {sol_energy/total_energy:.1%}")
        results['irrotational_fraction'] = irrot_energy / total_energy
        results['solenoidal_fraction'] = sol_energy / total_energy
        
        # Radial power spectrum (isotropic structure)
        k_mag = np.sqrt(kx**2 + ky**2)
        n_bins = N // 2
        radial_bins = np.linspace(0, k_mag.max(), n_bins)
        radial_power = np.zeros(n_bins - 1)
        total_power = np.abs(dx_fft)**2 + np.abs(dy_fft)**2
        
        for i in range(n_bins - 1):
            mask = (k_mag >= radial_bins[i]) & (k_mag < radial_bins[i+1])
            if mask.any():
                radial_power[i] = total_power[mask].mean()
        
        ax = axes[1, 1]
        ax.set_facecolor('#0d1117')
        ax.plot(radial_bins[:-1], radial_power, color='#44aaff', lw=2)
        ax.set_xlabel('|k| (spatial frequency)')
        ax.set_ylabel('Power')
        ax.set_title('Radial Power Spectrum (isotropy check)', color='#c0d8e8')
        ax.set_yscale('log')
        
        # Power law fit? (turbulence-like cascades)
        valid = radial_power > 0
        if valid.sum() > 5:
            log_k = np.log(radial_bins[:-1][valid] + 1e-10)
            log_p = np.log(radial_power[valid])
            slope, intercept = np.polyfit(log_k[1:], log_p[1:], 1)
            print(f"  Power law exponent: {slope:.2f} (k^α, turbulence: α≈-5/3)")
            results['power_law_exponent'] = slope
            ax.plot(radial_bins[:-1][valid], 
                   np.exp(intercept) * radial_bins[:-1][valid]**slope,
                   '--', color='#ff4444', lw=1, label=f'k^{slope:.1f}')
            ax.legend(facecolor='#0d1117', labelcolor='#ccc')
        
        # Vorticity field (real-space curl)
        curl_real = np.real(np.fft.ifft2(curl_fft))
        ax = axes[1, 2]
        ax.set_facecolor('#0d1117')
        vmax = np.abs(curl_real).max()
        im = ax.imshow(curl_real, cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                      aspect='auto', origin='lower')
        ax.set_title('Vorticity Field (real-space curl)', color='#c0d8e8')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Vector Field Spectral Decomposition — {basename}', 
                     color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_vector_field_spectral.png', 
                    dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════
    # E. Singular value distribution analysis
    # ═══════════════════════════════════════════════════════════════
    if 'singular_values' in data:
        sv = data['singular_values']
        if isinstance(sv, np.ndarray) and len(sv) > 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0a1a')
            
            # Marchenko-Pastur comparison (random matrix theory)
            # If SVs follow MP law → no structure; deviations = structure
            ax = axes[0]
            ax.set_facecolor('#0d1117')
            ax.hist(sv**2, bins=30, density=True, color='#44aaff', alpha=0.7,
                   label='Observed λ²')
            
            # MP law for comparison
            D = len(sv)
            gamma = 1.0  # aspect ratio (assuming square-ish)
            lambda_plus = (1 + np.sqrt(gamma))**2
            lambda_minus = (1 - np.sqrt(gamma))**2
            x = np.linspace(lambda_minus * sv.mean()**2, 
                           lambda_plus * sv.mean()**2, 100)
            mp_density = np.sqrt(np.maximum(0, (lambda_plus*sv.mean()**2 - x) * 
                                           (x - lambda_minus*sv.mean()**2))) / (2*np.pi*gamma*sv.mean()**2*x + 1e-10)
            ax.plot(x, mp_density, '--', color='#ff4444', lw=2, label='MP law (random)')
            ax.set_title('SV² vs Marchenko-Pastur', color='#c0d8e8')
            ax.legend(facecolor='#0d1117', labelcolor='#ccc')
            
            # Participation ratio (how many SVs contribute)
            sv_norm = sv / sv.sum()
            participation_ratio = 1.0 / np.sum(sv_norm**2)
            print(f"  Participation ratio: {participation_ratio:.1f}/{len(sv)} "
                  f"({participation_ratio/len(sv):.1%} of dimensions active)")
            results['participation_ratio'] = participation_ratio
            
            # Log-spacing of SVs (reveals hierarchical structure)
            ax = axes[1]
            ax.set_facecolor('#0d1117')
            log_sv = np.log(sv + 1e-10)
            log_sv_diffs = np.diff(log_sv)
            ax.plot(log_sv_diffs, 'o-', color='#ffaa44', markersize=3, lw=1)
            ax.axhline(np.mean(log_sv_diffs), color='#ff4444', ls='--', lw=1)
            ax.set_title('Log-SV Spacing (uniform = exponential decay)', color='#c0d8e8')
            ax.set_xlabel('Index')
            
            # Entropy of SV distribution
            sv_entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
            max_entropy = np.log(len(sv))
            print(f"  SV entropy: {sv_entropy:.2f} / {max_entropy:.2f} "
                  f"({sv_entropy/max_entropy:.1%} of maximum)")
            results['sv_entropy_ratio'] = sv_entropy / max_entropy
            
            # Effective dimension via different criteria
            ax = axes[2]
            ax.set_facecolor('#0d1117')
            cumulative = np.cumsum(sv**2) / np.sum(sv**2)
            ax.plot(cumulative, 'o-', color='#44ffaa', markersize=3, lw=1.5)
            for thresh in [0.5, 0.9, 0.95, 0.99]:
                eff_dim = np.searchsorted(cumulative, thresh) + 1
                ax.axhline(thresh, color='#555', ls=':', lw=0.5)
                ax.axvline(eff_dim, color='#555', ls=':', lw=0.5)
                ax.text(eff_dim + 0.5, thresh - 0.03, f'd={eff_dim}', 
                       color='#ccc', fontsize=7)
            ax.set_title('Cumulative Energy & Effective Dimensions', color='#c0d8e8')
            ax.set_xlabel('# dimensions')
            ax.set_ylabel('Fraction of variance')
            
            plt.suptitle(f'Singular Value Structure — {basename}', 
                         color='#c0d8e8', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{basename}_sv_structure.png', 
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
    
    return results


def cross_file_spectral_analysis(data_dir, output_dir="spectral_analysis"):
    """Compare spectral properties across steps/layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(data_dir, "jacobi_step*_layer*.npz")))
    if not files:
        print(f"No files found in {data_dir}")
        return
    
    print(f"Found {len(files)} files")
    
    # Collect spectral features across all files
    all_results = []
    for f in files:
        step_match = re.search(r'step(\d+)', f)
        layer_match = re.search(r'layer(\d+)', f)
        step = int(step_match.group(1)) if step_match else 0
        layer = int(layer_match.group(1)) if layer_match else 0
        
        r = explore_spectral_structures(f, output_dir)
        r['step'] = step
        r['layer'] = layer
        all_results.append(r)
    
    # ── Cross-file comparison plots ─────────────────────────────────
    if len(all_results) > 1:
        steps = sorted(set(r['step'] for r in all_results))
        layers = sorted(set(r['layer'] for r in all_results))
        
        # Heatmap of spectral gap across step × layer
        if any('h_out_2d_spectral_gap' in r for r in all_results):
            gap_matrix = np.zeros((len(steps), len(layers)))
            for r in all_results:
                si = steps.index(r['step'])
                li = layers.index(r['layer'])
                gap_matrix[si, li] = r.get('h_out_2d_spectral_gap', 0)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
            ax.set_facecolor('#0d1117')
            im = ax.imshow(gap_matrix.T, aspect='auto', cmap='viridis', 
                          origin='lower')
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer')
            ax.set_title('Graph Spectral Gap (step × layer)\nLarger = more clustered',
                        color='#c0d8e8')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cross_file_spectral_gap.png', 
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
        
        # Heatmap of irrotational/solenoidal fraction
        if any('irrotational_fraction' in r for r in all_results):
            irrot_matrix = np.zeros((len(steps), len(layers)))
            for r in all_results:
                si = steps.index(r['step'])
                li = layers.index(r['layer'])
                irrot_matrix[si, li] = r.get('irrotational_fraction', 0.5)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
            ax.set_facecolor('#0d1117')
            im = ax.imshow(irrot_matrix.T, aspect='auto', cmap='RdYlBu_r',
                          vmin=0, vmax=1, origin='lower')
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer')
            ax.set_title('Irrotational Fraction (step × layer)\n'
                        '1.0 = pure gradient flow, 0.0 = pure rotation',
                        color='#c0d8e8')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cross_file_helmholtz.png',
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
        
        # Participation ratio evolution
        if any('participation_ratio' in r for r in all_results):
            pr_matrix = np.zeros((len(steps), len(layers)))
            for r in all_results:
                si = steps.index(r['step'])
                li = layers.index(r['layer'])
                pr_matrix[si, li] = r.get('participation_ratio', 0)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
            ax.set_facecolor('#0d1117')
            im = ax.imshow(pr_matrix.T, aspect='auto', cmap='viridis',
                          origin='lower')
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer')
            ax.set_title('SV Participation Ratio (step × layer)\n'
                        'Higher = more dimensions active',
                        color='#c0d8e8')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cross_file_participation.png',
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
        
        # Power law exponent evolution
        if any('power_law_exponent' in r for r in all_results):
            ple_matrix = np.full((len(steps), len(layers)), np.nan)
            for r in all_results:
                si = steps.index(r['step'])
                li = layers.index(r['layer'])
                ple_matrix[si, li] = r.get('power_law_exponent', np.nan)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a1a')
            ax.set_facecolor('#0d1117')
            im = ax.imshow(ple_matrix.T, aspect='auto', cmap='coolwarm',
                          origin='lower', vmin=-4, vmax=0)
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer')
            ax.set_title('Power Law Exponent of Deformation Field\n'
                        '≈ -5/3 = turbulent cascade, ≈ -2 = smooth',
                        color='#c0d8e8')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cross_file_power_law.png',
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
        
        # ── Summary report ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Spectral analysis complete!")
        print(f"Output directory: {output_dir}/")
        print(f"Files analyzed: {len(files)}")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# F. ADDITIONAL DEEP STRUCTURE PROBES
# ═══════════════════════════════════════════════════════════════════════════

def probe_hidden_encodings(filepath, output_dir="spectral_analysis"):
    """
    Go beyond FFT: try multiple transforms to see if the data
    encodes structures in non-obvious ways.
    
    Probes:
      1. Autocorrelation of per-token signals (periodic patterns)
      2. Mutual information between Jacobian rows/cols
      3. Non-negative matrix factorization of |J| (parts-based decomposition)
      4. Independent Component Analysis of the point cloud
      5. Recurrence plot of the singular value sequence
      6. Symbolic dynamics: discretize and look for grammar
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_npz(filepath)
    basename = os.path.basename(filepath).replace('.npz', '')
    
    print(f"\n{'='*60}")
    print(f"Deep structure probes: {basename}")
    print(f"{'='*60}")
    
    # ── Probe 1: Autocorrelation of per-token signals ───────────────
    for key in ['per_token_volume', 'per_token_rotation', 'per_token_shear_mag']:
        if key not in data:
            continue
        signal = data[key]
        if len(signal) < 5:
            continue
        
        # Normalized autocorrelation
        sig_centered = signal - signal.mean()
        autocorr = np.correlate(sig_centered, sig_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # positive lags only
        autocorr /= autocorr[0] + 1e-10  # normalize
        
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')
        ax.plot(autocorr[:len(signal)//2], color='#44aaff', lw=1.5)
        ax.axhline(0, color='#555', lw=0.5)
        ax.axhline(2/np.sqrt(len(signal)), color='#ff4444', ls='--', lw=0.8,
                   label='95% confidence')
        ax.axhline(-2/np.sqrt(len(signal)), color='#ff4444', ls='--', lw=0.8)
        ax.set_title(f'Autocorrelation of {key}', color='#c0d8e8')
        ax.set_xlabel('Lag (tokens)')
        ax.set_ylabel('Autocorrelation')
        ax.legend(facecolor='#0d1117', labelcolor='#ccc')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_{key}_autocorr.png',
                    dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
        
        # Detect significant periodic lags
        threshold = 2 / np.sqrt(len(signal))
        significant_lags = np.where(np.abs(autocorr[1:len(signal)//2]) > threshold)[0] + 1
        if len(significant_lags) > 0:
            print(f"  {key}: significant autocorrelation at lags {significant_lags[:10]}")
    
    # ── Probe 2: NMF of |Jacobian| (parts-based decomposition) ─────
    if 'jacobian' in data:
        J = data['jacobian']
        J_abs = np.abs(J)
        
        # Ensure non-negative
        from sklearn.decomposition import NMF
        
        n_components_list = [2, 3, 5]
        fig, axes = plt.subplots(1, len(n_components_list), 
                                 figsize=(6*len(n_components_list), 5),
                                 facecolor='#0a0a1a')
        
        for idx, n_comp in enumerate(n_components_list):
            ax = axes[idx]
            ax.set_facecolor('#0d1117')
            
            try:
                nmf = NMF(n_components=n_comp, max_iter=500, random_state=42)
                W = nmf.fit_transform(J_abs)  # (D, n_comp) — basis activations
                H = nmf.components_            # (n_comp, D) — basis vectors
                
                # Plot the basis vectors as a heatmap
                im = ax.imshow(H, cmap='magma', aspect='auto')
                ax.set_title(f'NMF Components (k={n_comp})\n'
                            f'recon_error={nmf.reconstruction_err_:.2f}',
                            color='#c0d8e8', fontsize=9)
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Component')
                plt.colorbar(im, ax=ax)
                
                print(f"  NMF k={n_comp}: reconstruction error = "
                      f"{nmf.reconstruction_err_:.4f}")
            except Exception as e:
                ax.text(0.5, 0.5, f'NMF failed:\n{e}', transform=ax.transAxes,
                       ha='center', va='center', color='#ff4444', fontsize=8)
        
        plt.suptitle(f'Non-negative Matrix Factorization of |J| — {basename}',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_nmf.png',
                    dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()
    
    # ── Probe 3: Recurrence plot of singular values ─────────────────
    if 'singular_values' in data:
        sv = data['singular_values']
        if isinstance(sv, np.ndarray) and len(sv) > 5:
            # Recurrence plot: R[i,j] = 1 if |sv[i] - sv[j]| < threshold
            N = len(sv)
            diff_matrix = np.abs(sv[:, None] - sv[None, :])
            threshold = np.percentile(diff_matrix[diff_matrix > 0], 10)
            recurrence = (diff_matrix < threshold).astype(float)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a1a')
            
            ax = axes[0]
            ax.set_facecolor('#0d1117')
            ax.imshow(recurrence, cmap='binary', origin='lower', aspect='auto')
            ax.set_title('Recurrence Plot of Singular Values', color='#c0d8e8')
            ax.set_xlabel('SV Index i')
            ax.set_ylabel('SV Index j')
            
            # Recurrence quantification: diagonal line lengths
            # (indicates deterministic structure)
            diag_lengths = []
            for offset in range(1, N):
                diag = np.diag(recurrence, k=offset)
                current_len = 0
                for val in diag:
                    if val > 0.5:
                        current_len += 1
                    else:
                        if current_len > 1:
                            diag_lengths.append(current_len)
                        current_len = 0
                if current_len > 1:
                    diag_lengths.append(current_len)
            
            ax2 = axes[1]
            ax2.set_facecolor('#0d1117')
            if diag_lengths:
                ax2.hist(diag_lengths, bins=range(2, max(diag_lengths)+2),
                        color='#44ffaa', alpha=0.8, edgecolor='#2a3a4a')
                print(f"  Recurrence: {len(diag_lengths)} diagonal lines, "
                      f"max length = {max(diag_lengths)}")
            ax2.set_title('Diagonal Line Length Distribution\n'
                         '(longer = more deterministic structure)',
                         color='#c0d8e8', fontsize=9)
            ax2.set_xlabel('Line Length')
            ax2.set_ylabel('Count')
            
            plt.suptitle(f'Recurrence Analysis — {basename}',
                         color='#c0d8e8', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{basename}_recurrence.png',
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
    
    # ── Probe 4: ICA of the point cloud ─────────────────────────────
    for cloud_key in ['h_out_2d', 'h_in_2d']:
        if cloud_key not in data:
            continue
        points = data[cloud_key]
        if points.shape[0] < 10 or points.shape[1] < 2:
            continue
        
        try:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=2, random_state=42, max_iter=500)
            S = ica.fit_transform(points)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0a1a')
            
            ax = axes[0]
            ax.set_facecolor('#0d1117')
            ax.scatter(points[:, 0], points[:, 1], s=15, c='#44aaff', alpha=0.7)
            ax.set_title(f'{cloud_key} (original PCA)', color='#c0d8e8')
            
            ax = axes[1]
            ax.set_facecolor('#0d1117')
            ax.scatter(S[:, 0], S[:, 1], s=15, c='#ff44aa', alpha=0.7)
            ax.set_title(f'{cloud_key} (ICA components)', color='#c0d8e8')
            
            # Test for non-Gaussianity (kurtosis)
            from scipy.stats import kurtosis
            k0 = kurtosis(S[:, 0])
            k1 = kurtosis(S[:, 1])
            
            ax = axes[2]
            ax.set_facecolor('#0d1117')
            ax.hist(S[:, 0], bins=30, alpha=0.6, color='#ff44aa', 
                   label=f'IC1 (kurt={k0:.2f})', density=True)
            ax.hist(S[:, 1], bins=30, alpha=0.6, color='#44ffaa',
                   label=f'IC2 (kurt={k1:.2f})', density=True)
            ax.set_title('IC Distributions\n(high |kurtosis| = non-Gaussian = structured)',
                        color='#c0d8e8', fontsize=9)
            ax.legend(facecolor='#0d1117', labelcolor='#ccc', fontsize=8)
            
            print(f"  ICA {cloud_key}: kurtosis = [{k0:.2f}, {k1:.2f}] "
                  f"(|k|>1 suggests hidden structure)")
            
            plt.suptitle(f'ICA Decomposition — {basename} ({cloud_key})',
                         color='#c0d8e8', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{basename}_{cloud_key}_ica.png',
                        dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ICA failed for {cloud_key}: {e}")
    
    # ── Probe 5: Symbolic dynamics of the Jacobian ──────────────────
    if 'jacobian' in data:
        J = data['jacobian']
        
        # Discretize: positive → '+', negative → '-', near-zero → '0'
        threshold = np.std(J) * 0.1
        symbolic = np.where(J > threshold, '+', np.where(J < -threshold, '-', '0'))
        
        # Count bigram frequencies (2-symbol patterns)
        D = J.shape[0]
        bigrams = {}
        for i in range(D):
            for j in range(D - 1):
                bg = symbolic[i, j] + symbolic[i, j+1]
                bigrams[bg] = bigrams.get(bg, 0) + 1
        
        # Entropy of bigram distribution
        total = sum(bigrams.values())
        bg_probs = np.array(list(bigrams.values())) / total
        bg_entropy = -np.sum(bg_probs * np.log2(bg_probs + 1e-10))
        max_entropy = np.log2(len(bigrams))
        
        print(f"  Symbolic dynamics: {len(bigrams)} unique bigrams, "
              f"entropy = {bg_entropy:.2f} / {max_entropy:.2f} bits "
              f"({bg_entropy/max_entropy:.1%} of maximum)")
        
        if bg_entropy / max_entropy < 0.7:
            print(f"  → LOW entropy suggests structured/grammatical patterns in J!")
        
        # Visualize symbolic matrix
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a1a')
        
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        sym_numeric = np.where(symbolic == '+', 1, np.where(symbolic == '-', -1, 0)).astype(float)
        ax.imshow(sym_numeric, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('Symbolic Jacobian (+/0/-)', color='#c0d8e8')
        
        ax = axes[1]
        ax.set_facecolor('#0d1117')
        sorted_bg = sorted(bigrams.items(), key=lambda x: -x[1])[:20]
        names = [bg for bg, _ in sorted_bg]
        counts = [c for _, c in sorted_bg]
        ax.barh(range(len(names)), counts, color='#44aaff', alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color='#ccc')
        ax.set_title(f'Top Bigrams (entropy ratio: {bg_entropy/max_entropy:.1%})',
                    color='#c0d8e8', fontsize=9)
        ax.set_xlabel('Count')
        
        plt.suptitle(f'Symbolic Dynamics of Jacobian — {basename}',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{basename}_symbolic.png',
                    dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Spectral & hidden structure explorer for Jacobi NPZ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python3 spectral_explorer.py runs/0/jacobi_data/jacobi_step000100_layer02.npz

  # Analyze all files in a directory (cross-file comparison)
  python3 spectral_explorer.py runs/0/jacobi_data/

  # Deep structure probes on a single file
  python3 spectral_explorer.py runs/0/jacobi_data/jacobi_step000100_layer02.npz --deep

  # Everything
  python3 spectral_explorer.py runs/0/jacobi_data/ --deep --output-dir my_analysis
        """,
    )
    
    parser.add_argument("path", type=str,
                        help="Path to a single .npz file or directory of .npz files")
    parser.add_argument("--output-dir", type=str, default="spectral_analysis",
                        help="Output directory for plots")
    parser.add_argument("--deep", action="store_true",
                        help="Run deep structure probes (NMF, ICA, recurrence, symbolic)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file
        print(f"Analyzing single file: {args.path}")
        explore_spectral_structures(args.path, args.output_dir)
        if args.deep:
            probe_hidden_encodings(args.path, args.output_dir)
    elif os.path.isdir(args.path):
        # Directory — cross-file analysis
        print(f"Analyzing directory: {args.path}")
        cross_file_spectral_analysis(args.path, args.output_dir)
        
        # Also run deep probes on a representative subset
        if args.deep:
            import glob as _glob
            files = sorted(_glob.glob(os.path.join(args.path, "jacobi_step*_layer*.npz")))
            # Pick ~5 representative files (first, last, middle, etc.)
            if files:
                indices = set()
                indices.add(0)
                indices.add(len(files) - 1)
                indices.add(len(files) // 2)
                indices.add(len(files) // 4)
                indices.add(3 * len(files) // 4)
                for idx in sorted(indices):
                    if idx < len(files):
                        print(f"\n--- Deep probes on: {os.path.basename(files[idx])} ---")
                        probe_hidden_encodings(files[idx], args.output_dir)
    else:
        print(f"Error: {args.path} not found")
        sys.exit(1)
    
    n_plots = len(glob.glob(os.path.join(args.output_dir, '*.png')))
    print(f"\n{'='*60}")
    print(f"Done! Generated {n_plots} visualizations in {args.output_dir}/")
    print(f"{'='*60}")
