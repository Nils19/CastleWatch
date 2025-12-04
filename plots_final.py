import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import stft


import load_specs.py
import filters.py
import frequency_analyzer.py





#BALDUR 

def plot_waveform(wave, wave_type):
    t = np.arange(len(wave))
    plt.figure(figsize=(12, 4))
    plt.plot(t, wave)
    plt.title(f"{wave_type.capitalize()} Wave (summed)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Summed_{wave_type}_Waveform.pdf")
    plt.show()




# PLOTTING SECTION (HEINI)


def plot_single_frame_fft(x, fs, window, window_len):
    """
    Plot the magnitude spectrum of one windowed frame.
    """
    frame = x[:window_len]
    windowed_frame = frame * window

    # FFT
    spectrum = np.fft.fft(windowed_frame)
    half = window_len // 2
    freqs = np.fft.fftfreq(window_len, 1/fs)[:half]
    mag = np.abs(spectrum[:half])

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag)
    plt.title("Single-Frame FFT Magnitude Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.savefig("Single_Frame_FFT_Magnitude_Spectrum.pdf")
    plt.show()


def plot_spectrogram(f, t, mag_db, fs):
    """
    Plot STFT spectrogram (frequency vs. time).
    """
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, mag_db, shading='gouraud', vmin=-120, vmax=0)
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("STFT Spectrogram")
    plt.ylim(0, fs / 2)
    plt.savefig("STFT_Spectrogram.pdf")
    plt.show()


def plot_peak_tracking(t, peak_freqs):
    """
    Plot dominant frequency over time.
    """
    plt.figure(figsize=(10, 3))
    plt.plot(t, peak_freqs, color='gold')
    plt.title("Dominant Frequency Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.grid()
    plt.savefig("Dominant_Frequency_Over_Time.pdf")
    plt.show()
    
def plot_real_imaginary(f, t):
    """
    Plot real and imaginary parts of the STFT.
    """
    plt.figure(figsize=(12, 6))

    # Real part
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, np.real(Zxx), shading='gouraud')
    plt.colorbar(label="Real Part")
    plt.title("Real Part of STFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, fs / 2)

    # Imaginary part
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f, np.imag(Zxx), shading='gouraud')
    plt.colorbar(label="Imaginary Part")
    plt.title("Imaginary Part of STFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, fs / 2)

    plt.tight_layout()
    plt.savefig("Real_Imaginary_Parts_of_STFT.pdf")
    plt.show()
    


# ------------------------------------------------------------
# CALL ALL PLOTS
# ------------------------------------------------------------

# Create window for single-frame FFT
window = create_window(window_len, window_type)

# 1) Single-frame FFT plot
plot_single_frame_fft(data, fs, window, window_len)

# 2) Spectrogram plot
plot_spectrogram(f, t, mag_db, fs)

# 3) Peak frequency tracking plot
plot_peak_tracking(t, peak_freqs)

#4) Real and Imaginary parts plot
plot_real_imaginary(f,t)




# EKKO SECTION (FILTER PLOTTING)
def fir_or_iir(specs):
    structure = specs["filter_structure"].lower()

    if structure == 'fir':
        h = firFilter(specs)
        b = h.copy()
        a = np.array([1.0])
        
        return structure, b, a

    elif structure == 'iir':
        z_norm, p_norm = prototype_picker(specs)
        b, a = iirFilter(specs, z_norm, p_norm)
        return structure, b, a

    else:
        raise ValueError("filter_structure must be 'fir' or 'iir'")



#structure, b, a = fir_or_iir(specs)
#h = b

def plot_magnitude_phase_response(b, a, structure):
    w, H = signal.freqz(b, a, worN=2048, fs=specs["fs_filter"])

    plt.figure()
    plt.plot(w, 20*np.log10(np.maximum(np.abs(H), 1e-12)))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title(f"{structure.upper()} magnitude response")
    plt.grid(True)
    plt.savefig(f"Magnitude_Response_Of_{structure.upper()}.pdf")
    plt.show()

    plt.figure()
    plt.plot(w, np.unwrap(np.angle(H)))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.title(f"{structure.upper()} phase response")
    plt.savefig(f"Phase_Response_Of_{structure.upper()}.pdf")
    plt.grid(True)
    plt.show()
    
    
def plot_pz(b, a, structure):
    z_z = np.roots(b)
    p_z = np.roots(a)

    plt.figure()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.scatter(np.real(p_z), np.imag(p_z), marker='x', label='poles')
    plt.scatter(np.real(z_z), np.imag(z_z), marker='o',
                facecolors='none', edgecolors='C1', label='zeros')

    # unit circle for reference
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), linestyle='--', linewidth=0.7)

    plt.xlabel('real(z)')
    plt.ylabel('imag(z)')
    plt.title(f"{structure.upper()} digital filter poles and zeros")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(f"Pole_Zero_Plot_of_{structure.upper()}_digital_filter.pdf")
    plt.show()



# impulse response
def impulse_response(b, fs):
    t = np.arange(len(b)) / fs
    plt.figure()    
    plt.stem(t, b)
    plt.xlabel("Time [s]")
    plt.ylabel("h[n]")
    plt.title(f"{structure.upper()} impulse response")
    plt.savefig(f"Impulse_Response_Of_{structure.upper()}.pdf")
    plt.grid(True)
    plt.show()
    
plot_magnitude_phase_response(b,a,structure)
plot_pz(b,a,structure)
impulse_response(b, specs["fs_filter"])



def save_filter_coeffs(filter_info, prefix="filter"):
    struct = filter_info["structure"]
    b      = np.asarray(filter_info["b"], dtype=float)
    a      = np.asarray(filter_info["a"], dtype=float)

    # One simple text file with everything
    fname = f"{prefix}_coeffs_{struct}.txt"
    with open(fname, "w") as f:
        f.write(f"# filter structure: {struct}\n")
        f.write("# b-coefficients (feedforward):\n")
        np.savetxt(f, b[None, :], fmt="%.16e")  # one line

        f.write("# a-coefficients (feedback):\n")
        np.savetxt(f, a[None, :], fmt="%.16e")  # one line

    return fname

save_filter_coeffs({"structure": structure, "b": b, "a": a})





