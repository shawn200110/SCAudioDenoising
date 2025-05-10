import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import toolbox as tb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter


def denoise_via_spectrogram(signal, noisy_signal, fs, n_fft=1024, plot_out=True):
    Z_X_raw = tb.compute_stft(signal, fs, n_fft, plot_out=False)
    Z_X, t, f = tb.compute_stft(noisy_signal, fs, n_fft, plot_out=False)

    original_phase = np.angle(Z_X)


    U, S, Vt = np.linalg.svd(np.abs(Z_X), full_matrices=False)

    # S[2:len(S)] = 0

    # 2. Choose rank k (number of components to keep)
    k = 1  # or 2, or 5, depending on how much you want to keep

    # 3. Keep only the top-k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # 4. Reconstruct the low-rank approximation
    spectrogram_matrix_reconstructed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))

    # Step 2: Combine reconstructed magnitude with original phase
    Zxx_reconstructed = spectrogram_matrix_reconstructed * np.exp(1j * original_phase)

    reconstructed_signal = tb.compute_istft(Zxx_reconstructed, fs, n_fft, plot_out=True)
    sf.write('reconstructed.wav', reconstructed_signal, fs)

    if plot_out:
        plt.subplot(3,1,1)
        plt.plot(signal)
        plt.subplot(3,1,2)
        plt.plot(noisy_signal)
        plt.subplot(3,1,3)
        plt.plot(reconstructed_signal)
        plt.show()
    return 1

def dtwf_ece513(signal, noisy_signal, fs, A, plot_out=True):
    """
    Denoise using a model-based projection approach: y = Ax + n

    Args:
        signal (np.ndarray): Ground truth clean signal (for plotting).
        noisy_signal (np.ndarray): Observed noisy signal.
        fs (int): Sampling rate.
        A (np.ndarray): Known linear mixing matrix (shape: m x n).
        plot_out (bool): Whether to show plots.

    Returns:
        xhat (np.ndarray): Estimated clean signal.
    """
    # Step 1: Reshape and construct matrix Y from noisy signal
    

    A = np.eye(np.shape(noisy_signal)[0])
    m, n = A.shape
    assert len(noisy_signal) >= m, "Signal too short for number of rows in A"

    # Truncate signal to fit reshape if needed
    N = len(noisy_signal) - (len(noisy_signal) % m)
    y_matrix = noisy_signal[:N].reshape(m, -1)  # shape: m x T

    # Step 2: Estimate R_xx and R_yy
    Rxx = np.cov(y_matrix)  # assuming x and y covariance structure is similar
    Ryy = A @ Rxx @ A.T + 1e-6 * np.eye(m)  # Add small regularization for stability

    # Step 3: Estimate Rxy
    Rxy = Rxx @ A.T

    # Step 4: Solve for optimal linear estimator
    H = Rxy @ np.linalg.inv(Ryy)  # shape: n x m

    # Step 5: Estimate clean signal
    x_matrix_hat = H @ y_matrix  # shape: n x T

    # Step 6: Flatten output
    xhat = x_matrix_hat.flatten()

    # Save reconstructed
    sf.write('model_projected.wav', xhat.astype(np.float32), fs)

    if plot_out:
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(signal[:len(xhat)], label='Clean Signal')
        plt.legend()
        # plt.subplot(3, 1, 2)
        # plt.plot(noisy_signal[:len(xhat)], label='Noisy Signal')
        # plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(xhat, label='Estimated Signal (Model-Based)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return xhat


def denoise_via_2dimPCA(signal, noisy_signal, pc_scale_factor, fs, k, plot_out=True):
    X = np.vstack((noisy_signal[:-k], noisy_signal[k:]))  # Shape: (2, N-k)

    # === Step 4: Perform SVD ===
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # === Step 5: Project onto principal components ===
    PCs = np.dot(U.T, X)

    # === Step 6: Reconstruct using second PC only (hypothesis: cleaner signal)
    reconstructed1 = np.dot(U[:, 0:1], PCs[0:1, :])
    reconstructed2 = np.dot(U[:, 1:2], PCs[1:2, :])

    # Rebuild 1D signal (taking first row of reconstructed matrix)
    reconstructed_signal1 = reconstructed1[0, :]
    reconstructed_signal2 = reconstructed2[0, :]

    # === Step 7: Save the reconstructed audio ===
    sf.write('pca_component1.wav', reconstructed_signal1, fs)
    sf.write('pca_component2.wav', reconstructed_signal2, fs)

    xhat = tb.spectral_gate(reconstructed_signal1, pc_scale_factor*reconstructed_signal2, fs, n_fft=1024, hop=512)

    sf.write('final_reconstructed.wav', xhat, fs)

    if plot_out:
        plt.figure(figsize=(12, 8))
        plt.subplot(5, 1, 1)
        plt.plot(signal[:-k])
        plt.title('Clean Signal')
        plt.subplot(5, 1, 2)
        plt.plot(noisy_signal[:-k])
        plt.title('Noisy Signal')
        plt.subplot(5, 1, 3)
        plt.plot(reconstructed_signal1)
        plt.title('PC1 Signal')
        plt.subplot(5, 1, 4)
        plt.plot(reconstructed_signal2)
        plt.title('PC2 Signal')
        plt.subplot(5, 1, 5)
        plt.plot(xhat)
        plt.title('PC1 After Spectral Masking Using PC2')
        plt.xlabel("Sample #")
        plt.tight_layout()
        plt.show()

   

    return xhat,reconstructed_signal1, reconstructed_signal2



def dtwf(signal, noisy_signal, fs, L, noise_samples=[200, 1600], plot_out=True):
    N = len(noisy_signal)
    y = noisy_signal

    sigma_v2 = tb.estimate_noise_variance(noisy_signal, noise_samples[0], noise_samples[1])

    # Estimate autocorrelation of y(n)
    ry = np.correlate(y, y, mode='full')[N-1:N-1+L] / N
    Ry = toeplitz(ry)

    u1 = np.zeros(L)    # u1 vector (impulse vector)
    u1[0] = 1

    ho = u1 - sigma_v2 * np.linalg.solve(Ry, u1) # Compute Wiener filter h_o

    xhat = lfilter(ho, 1.0, y) # Apply filter to y(n)

    # Save to disk
    sf.write('dtwf_reconstruct.wav', xhat.astype(np.float32), fs)

    # Compute normalized MMSE estimate (using first tap)
    Ey = ry[0]  # total power in y(n), approximate
    Jx_tilde = ho[0]

    if plot_out:
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(signal)
        plt.legend()
        plt.title("Clean Signal")
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(noisy_signal)
        plt.legend()
        plt.title("Noisy Signal")
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(xhat)
        plt.legend()
        plt.title("Reconstructed Signal")
        plt.xlabel("Sample Index")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return xhat, Jx_tilde, ho, sigma_v2


filename = "ECE513_FemaleSpeechSample.wav"
# filename = "ECE513_FinalProject_Mixed.wav"
# filename = "ECE513_FinalProject-001.wav"
# filename2 = "ECE513_FinalProject-002.wav"

# Read In Signal
N = 200000
signal, fs = sf.read(filename)
if signal.ndim > 1:
    signal = np.mean(signal, axis=1)  # convert to mono if stereo
signal = signal[:N]

# signal2, fs = sf.read(filename2)
# if signal2.ndim > 1:
#     signal2 = np.mean(signal2, axis=1)  # convert to mono if stereo
# signal2 = signal2[:N]

# signal = signal + signal2

# Generate Noisy Signal
noise = tb.noise_generator(len(signal), 0.015, fs, "burst")
noisy_signal = signal+noise


# noisy_signal = signal

sf.write('noisy_signal.wav', noisy_signal, fs)
# Choose a method of denoising

# noisy_signal = np.vstack([noisy_signal,noisy_signal])

# dtwf_ece513(signal, noisy_signal, fs, 1)
# denoise_via_2dimPCA(signal, noisy_signal, 5,fs, 1)
# denoise_via_spectrogram(signal, noisy_signal, fs, n_fft=256)

ns = [13000,20000]#Elec Guitar
xhat,Jx_tilde, ho,sigma_v2 = dtwf(signal[:100000], noisy_signal[:100000], fs, 135)






## RUN DTWF Looper ####################################################################################
# # Range of filter lengths to test
# L_values = np.arange(5, 201, 20)
# # L_values = np.array([65]) # optimal filter length for the male speech
# L_values = np.array([135]) # optimal filter length for the female speech
# mse_values = []
# N=50000;


# for L in L_values:
#     xhat, Jx_tilde, ho, sigma_v2 =dtwf(signal[:N], noisy_signal[:N], 44100, L)
#     error = signal[L-1:N] - xhat[L-1:]
#     mse = np.mean(error**2)
#     mse_values.append(mse)

# # Plot MSE vs Filter Length
# plt.figure(figsize=(8, 4))
# plt.plot(L_values, mse_values, marker='o')
# plt.title("Filter Length vs Signal Estimation Error")
# plt.xlabel("Filter Length (L)")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(sigma_v2)

