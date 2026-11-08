import numpy as np
import matplotlib.pyplot as plt

def awokolas_law(x, P, D, gamma=1.5):
    """
    Computes the Attention Retrieval Probability (A) based on Awokola's Law.
    
    Parameters:
    x     : float or numpy array -> Normalized position in context window
    P     : float -> Model parameter scale (e.g., 7e9 for 7B, 70e9 for 70B)
    D     : float -> Textual distraction/noise factor (D >= 1)
    gamma : float -> Systemic architectural fatigue constant
    """
    ln_P = np.log(P)
    position_term = 1.0 + 4.0 * (x - 0.5)**2
    exponent = -gamma / (D * ln_P * position_term)
    return np.exp(exponent)

# Generate a continuous spectrum of context positions from start (0) to end (1)
x_space = np.linspace(0, 1, 500)

# Simulate three structural scales under Awokola's Law
A_7B   = awokolas_law(x_space, P=7e9,   D=1.0)   # Small dense model
A_70B  = awokolas_law(x_space, P=70e9,  D=1.0)  # Mid-sized dense model
A_405B = awokolas_law(x_space, P=405e9, D=1.0) # Flagship scale model

# Plotting the empirical U-shape decay profiles
plt.figure(figsize=(10, 6))
plt.plot(x_space, A_7B,   label="Small Scale Model (7B parameters)",   color="#ff4d4d", lw=2)
plt.plot(x_space, A_70B,  label="Medium Scale Model (70B parameters)",  color="#ffa500", lw=2, linestyle="--")
plt.plot(x_space, A_405B, label="Flagship Scale Model (405B parameters)", color="#2ecc71", lw=2.5)

# Graph Formatting
plt.title("Awokola's Law: Context Window Attention Decay Curve", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Normalized Position of Target Information inside Prompt (x)", fontsize=11)
plt.ylabel("Attention Retrieval Probability (A)", fontsize=11)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower center", fontsize=10)

print("[Awokola's Law] Generating and displaying attention degradation simulation curves...")
plt.show()
