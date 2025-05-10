import numpy as np
import soundfile as sf
from scipy.linalg import toeplitz
from scipy.signal import butter, lfilter, stft, istft, filtfilt 
import matplotlib.pyplot as plt


def butter_band_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def estimate_autocorrelation(signal, max_lag):
    """Estimate biased autocorrelation up to lag max_lag."""
    N = len(signal)
    autocorr = np.correlate(signal, signal, mode='full') / N
    mid = len(autocorr) // 2
    return autocorr[mid:mid+max_lag]

def dtwf_filter(source_wav_path, output_wav_path, filter_order=50, epsilon=1e-6):
    """Apply Discrete-Time Wiener Filter (DTWF) based on LMV notes to denoise source.wav."""
    
    # Load noisy source
    y, fs = sf.read(source_wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # Convert to mono if stereo
    
    # Estimate autocorrelation R_yy
    r_yy = estimate_autocorrelation(y, filter_order)
    R_yy = toeplitz(r_yy)
    
    # Approximate cross-correlation R_xy
    var_y = np.var(y)
    r_xy = np.zeros(filter_order)
    r_xy[0] = var_y  # Assume signal and noise are uncorrelated; place var at lag 0
    
    # Regularize R_yy slightly for stability
    R_yy += epsilon * np.eye(filter_order)
    
    # Solve for K
    K = np.linalg.solve(R_yy, r_xy)
    
    # Filter the signal (linear convolution)
    y_hat = np.convolve(y, K, mode='same')
    
    # Save output
    sf.write(output_wav_path, y_hat, fs)
    
    return K


def compute_stft(signal, fs, n_fft, plot_out=True):
    hop_length = n_fft // 4  # 75% overlap (common choice)
    frequencies, times, Zxx = stft(signal, fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    spectrogram_matrix = np.abs(Zxx)  # Take the magnitude (ignore phase for now)
    print(f"Spectrogram shape: {spectrogram_matrix.shape}")  # (n_frequencies, n_time_frames)
    if plot_out:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, 20*np.log10(spectrogram_matrix + 1e-10), shading='gouraud')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Spectrogram (Magnitude)')
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.tight_layout()
        plt.show()
    return Zxx, times, frequencies

def compute_istft(X, fs, n_fft, plot_out=True):
    # Step 3: Inverse STFT
    hop_length = n_fft // 4  # 75% overlap (common choice)
    _, reconstructed_signal = istft(X, fs, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Step 4: Save the output
    # sf.write('reconstructed_audio.wav', reconstructed_signal, fs)
    return reconstructed_signal

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data)

def estimate_noise_variance(y, start_sample, end_sample):
    noise_segment = y[start_sample:end_sample]
    sigma_v2_est = np.var(noise_segment)
    return sigma_v2_est

def spectral_gate(pc1, pc2, fs, n_fft=1024, hop=512):
    f, t, P1 = stft(pc1, fs, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, P2 = stft(pc2, fs, nperseg=n_fft, noverlap=n_fft - hop)

    magnitude_ratio = np.abs(P2) / (np.abs(P1) + 1e-8)
    mask = 1 - magnitude_ratio
    mask = np.clip(mask, 0, 1)

    cleaned = mask * P1
    _, denoised = istft(cleaned, fs, nperseg=n_fft, noverlap=n_fft - hop)
    return denoised


def noise_generator(length, amplitude, fs, type='white'):
    if type == "white":
        noise = amplitude*np.random.normal(0, 1, size=length)
    elif type == "burst":
        noise = np.zeros(length)
        burst_length = int((10/ 1000.0) * fs)

        rng = np.random.default_rng(None)
        burst_positions = rng.choice(length - burst_length, 100, replace=False)

        for pos in burst_positions:
            noise[pos:pos + burst_length] += rng.normal(scale=amplitude, size=burst_length)
    elif type == "hiss":
        t = np.arange(length) / fs
        noise = 0.01 * np.sin(2 * np.pi * 3000 * t) + 0.01 * np.random.randn(len(t))   

    return noise
