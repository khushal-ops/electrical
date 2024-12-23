import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Generate a sample signal (sine wave + noise)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
original_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.normal(size=fs)

# Initial FIR filter size (window size)
initial_window_size = 10

def fir_filter(signal, window_size):
    """Apply a simple FIR filter using a moving average kernel."""
    kernel = np.ones(window_size) / window_size  # The filter coefficients (moving average)
    return np.convolve(signal, kernel, mode='same')  # Apply FIR filter

def constant_amplitude(signal, filtered_signal):
    """Ensure the filtered signal amplitude matches the original signal."""
    # Ensure the maximum of the filtered signal does not exceed the max of the original
    max_original = np.max(np.abs(signal))
    max_filtered = np.max(np.abs(filtered_signal))
    
    if max_filtered != 0:
        # Scale the filtered signal to match the maximum amplitude of the original signal
        return filtered_signal * (max_original / max_filtered)
    return filtered_signal

# Create the figure and two axes for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Original Signal vs FIR Filtered Signal", fontsize=16)

# Plot the original signal in the first subplot
ax1.set_title("Original Signal")
ax1.plot(t, original_signal, color="blue")
ax1.set_xlim(0, 1)
ax1.set_ylim(-2, 2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.grid()

# Apply FIR filter and adjust the amplitude to match
filtered_signal = fir_filter(original_signal, initial_window_size)
adjusted_signal = constant_amplitude(original_signal, filtered_signal)

# Plot the FIR filtered signal in the second subplot
ax2.set_title("FIR Filtered Signal")
filtered_line, = ax2.plot(t, adjusted_signal, color="red")
ax2.set_xlim(0, 1)
ax2.set_ylim(-2, 2)
ax2.set_xlabel("Time (s)")
ax2.grid()

# Add a slider to control the "volume" (smoothing level)
slider_ax = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor="lightgoldenrodyellow")
volume_slider = Slider(slider_ax, "Volume (Averaging Level)", 1, 100, valinit=initial_window_size, valstep=1)

def update(val):
    """Update the FIR filtered signal when the slider is adjusted."""
    window_size = int(volume_slider.val)
    new_filtered_signal = fir_filter(original_signal, window_size)
    adjusted_filtered_signal = constant_amplitude(original_signal, new_filtered_signal)
    filtered_line.set_ydata(adjusted_filtered_signal)
    fig.canvas.draw_idle()  # Redraw the figure

# Connect the slider to the update function
volume_slider.on_changed(update)

# Show the plot
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()